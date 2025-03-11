from dataclasses import dataclass, field
from typing import List
from sionna.nr.utils import generate_prng_seq
from sionna.nr import PUSCHConfig, CarrierConfig, PUSCHDMRSConfig, TBConfig, PUSCHPilotPattern, TBEncoder,LayerMapper, LayerDemapper, TBDecoder, PUSCHLSChannelEstimator
from sionna.channel import AWGN, OFDMChannel
from sionna.ofdm import LinearDetector, ResourceGrid, ResourceGridMapper
from sionna.mimo import StreamManagement
from sionna.mapping import Mapper
from sionna.utils import BinarySource

import tensorflow as tf
import numpy as np
import pickle
import h5py

from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization, SeparableConv2D
from tensorflow.nn import relu

from collections import namedtuple

@dataclass
class SystemConfig:
    NCellId: int = 246
    FrequencyRange: int = 1
    BandWidth: int = 100
    Numerology: int = 1
    CpType: int = 0
    NTxAnt: int = 1
    NRxAnt: int = 8
    BwpNRb: int = 273
    BwpRbOffset: int = 0
    harqProcFlag: int = 0
    nHarqProc: int = 1
    rvSeq: int = 0


@dataclass
class UeConfig:
    TransformPrecoding: int = 0
    Rnti: int = 20002
    nId: int = 246
    CodeBookBased: int = 0
    DmrsPortSetIdx: List[int] = field(default_factory=lambda: [0])  # FIXED
    NLayers: int = 1
    NumDmrsCdmGroupsWithoutData: int = 2
    Tpmi: int = 0
    FirstSymb: int = 0
    NPuschSymbAll: int = 14
    RaType: int = 1
    FirstPrb: int = 31
    NPrb: int = 4
    FrequencyHoppingMode: int = 0
    McsTable: int = 0
    Mcs: int = 3
    ILbrm: int = 0
    nScId: int = 0
    NnScIdId: int = 246
    DmrsConfigurationType: int = 0
    DmrsDuration: int = 1
    DmrsAdditionalPosition: int = 1
    PuschMappingType: int = 0
    DmrsTypeAPosition: int = 3
    HoppingMode: int = 0
    NRsId: int = 0
    Ptrs: int = 0
    ScalingFactor: int = 0
    OAck: int = 0
    IHarqAckOffset: int = 11
    OCsi1: int = 0
    ICsi1Offset: int = 7
    OCsi2: int = 0
    ICsi2Offset: int = 0
    NPrbOh: int = 0
    nCw: int = 1
    TpPi2Bpsk: int = 0

@dataclass
class MyConfig:
    Sys: SystemConfig
    Ue: List[UeConfig]
    Num_tx: int = 1
    Num_rx: int = 1
    Carrier_frequency: float = 2.55e9  # Carrier frequency in Hz

class MyPUSCHConfig(PUSCHConfig):
    def __init__(self, My_Config: MyConfig, slot_number=4, frame_number=0):
        # assert len(My_Config.Ue) == 1, "only suppport 1"
        assert My_Config.Ue[0].NLayers == 1
        self.My_Config = My_Config
        super().__init__(
            carrier_config=CarrierConfig(
                n_cell_id=My_Config.Sys.NCellId,
                cyclic_prefix="normal" if ~My_Config.Sys.CpType else "extended",
                subcarrier_spacing=15*(2**My_Config.Sys.Numerology),
                n_size_grid=My_Config.Sys.BwpNRb,
                n_start_grid=My_Config.Sys.BwpRbOffset,
                slot_number=slot_number,
                frame_number=frame_number
            ),
            pusch_dmrs_config=PUSCHDMRSConfig(
                config_type=My_Config.Ue[0].DmrsConfigurationType + 1,
                length=My_Config.Ue[0].DmrsDuration,
                additional_position=My_Config.Ue[0].DmrsAdditionalPosition,
                dmrs_port_set=My_Config.Ue[0].DmrsPortSetIdx,
                n_id=My_Config.Ue[0].NnScIdId,
                n_scid=My_Config.Ue[0].nScId,
                num_cdm_groups_without_data=My_Config.Ue[0].NumDmrsCdmGroupsWithoutData,
                type_a_position=My_Config.Ue[0].DmrsTypeAPosition
            ),
            tb_config=TBConfig(
                channel_type='PUSCH',
                n_id=My_Config.Ue[0].nId,
                mcs_table=My_Config.Ue[0].McsTable + 1,
                mcs_index=My_Config.Ue[0].Mcs
            ),
            mapping_type='A' if ~My_Config.Ue[0].PuschMappingType else 'B',
            n_size_bwp=My_Config.Sys.BwpNRb,
            n_start_bwp=My_Config.Sys.BwpRbOffset,
            num_layers=My_Config.Ue[0].NLayers,
            num_antenna_ports=len(My_Config.Ue[0].DmrsPortSetIdx),
            precoding='non-codebook' if ~My_Config.Ue[0].CodeBookBased else 'codebook',
            tpmi=My_Config.Ue[0].Tpmi,
            transform_precoding=False if ~My_Config.Ue[0].TransformPrecoding else True,
            n_rnti=My_Config.Ue[0].Rnti,
            symbol_allocation=[My_Config.Ue[0].FirstSymb,My_Config.Ue[0].NPuschSymbAll]
        )

    @property
    def phy_cell_id(self):
        return self._carrier._n_cell_id
    
    @phy_cell_id.setter
    def phy_cell_id(self, value):
        self.carrier._n_cell_id = value
        self.tb._n_id = value
        self.dmrs._n_id = value

    @property
    def first_resource_block(self):
        """
        :class:`~sionna.nr.CarrierConfig` : Carrier configuration
        """
        return self.My_Config.Ue[0].FirstPrb

    @property
    def first_subcarrier(self):
        """
        :class:`~sionna.nr.CarrierConfig` : Carrier configuration
        """
        return 12*self.first_resource_block

    @property
    def num_resource_blocks(self):
        """
        int, read-only : Number of allocated resource blocks for the
            PUSCH transmissions.
        """
        return self.My_Config.Ue[0].NPrb

    @property
    def dmrs_grid(self):
        # pylint: disable=line-too-long
        """
        complex, [num_dmrs_ports, num_subcarriers, num_symbols_per_slot], read-only : Empty
            resource grid for each DMRS port, filled with DMRS signals

            This property returns for each configured DMRS port an empty
            resource grid filled with DMRS signals as defined in
            Section 6.4.1.1 [3GPP38211]. Not all possible options are implemented,
            e.g., frequency hopping and transform precoding are not available.

            This property provides the *unprecoded* DMRS for each configured DMRS port.
            Precoding might be applied to map the DMRS to the antenna ports. However,
            in this case, the number of DMRS ports cannot be larger than the number of
            layers.
        """
        # Check configuration
        self.check_config()

        # Configure DMRS ports set if it has not been set
        reset_dmrs_port_set = False
        if len(self.dmrs.dmrs_port_set)==0:
            self.dmrs.dmrs_port_set = list(range(self.num_layers))
            reset_dmrs_port_set = True

        # Generate empty resource grid for each port
        a_tilde = np.zeros([len(self.dmrs.dmrs_port_set),
                            self.num_subcarriers,
                            self.carrier.num_symbols_per_slot],
                            dtype=complex)
        first_subcarrier = self.first_subcarrier
        num_subcarriers = self.num_subcarriers

        # For every l_bar
        for l_bar in self.l_bar:

            # For every l_prime
            for l_prime in self.l_prime:

                # Compute c_init
                l = l_bar + l_prime
                c_init = self.c_init(l)
                # Generate RNG
                c = generate_prng_seq(first_subcarrier + num_subcarriers, c_init=c_init)
                c = c[first_subcarrier:]

                # Map to QAM
                r = 1/np.sqrt(2)*((1-2*c[::2]) + 1j*(1-2*c[1::2]))

                # For every port in the dmrs port set
                for j_ind, _ in enumerate(self.dmrs.dmrs_port_set):

                    # For every n
                    for n in self.n:

                        # For every k_prime
                        for k_prime in [0, 1]:

                            if self.dmrs.config_type==1:
                                k = 4*n + 2*k_prime + \
                                    self.dmrs.deltas[j_ind]
                            else: # config_type == 2
                                k = 6*n + k_prime + \
                                    self.dmrs.deltas[j_ind]

                            a_tilde[j_ind, k, self.l_ref+l] = \
                                r[2*n + k_prime] * \
                                self.dmrs.w_f[k_prime][j_ind] * \
                                self.dmrs.w_t[l_prime][j_ind]

        # Amplitude scaling
        a = self.dmrs.beta*a_tilde

        # Reset DMRS port set if it was not set
        if reset_dmrs_port_set:
            self.dmrs.dmrs_port_set = []

        return a

class MySimulator():
    def __init__(self, pusch_config: MyPUSCHConfig):

        self.Num_rx = pusch_config.My_Config.Num_rx
        self.Num_tx = pusch_config.My_Config.Num_tx
    
        tb_size = pusch_config.tb_size
        num_coded_bits = pusch_config.num_coded_bits
        target_coderate = pusch_config.tb.target_coderate
        num_bits_per_symbol = pusch_config.tb.num_bits_per_symbol

        num_layers = pusch_config.num_layers
        n_rnti = pusch_config.n_rnti
        n_id = pusch_config.tb.n_id

        self.Binary_Source = BinarySource(dtype=tf.float32)
        self.TB_Encoder = TBEncoder(target_tb_size=tb_size,
                            num_coded_bits=num_coded_bits,
                            target_coderate=target_coderate,
                            num_bits_per_symbol=num_bits_per_symbol,
                            num_layers=num_layers,
                            n_rnti=n_rnti,
                            n_id=n_id,
                            channel_type="PUSCH",
                            codeword_index=0,
                            use_scrambler=True,
                            verbose=False,
                            output_dtype=tf.float32)
        
        self.Constellation_Mapper = Mapper("qam", num_bits_per_symbol, dtype=tf.complex64)

        self.Layer_Mapper = LayerMapper(num_layers=num_layers, dtype=tf.complex64)
    
        self.Pilot_Pattern = PUSCHPilotPattern([pusch_config], dtype=tf.complex64)

        num_subcarriers = pusch_config.num_subcarriers
        subcarrier_spacing = pusch_config.carrier.subcarrier_spacing*1e3
        fft_size = num_subcarriers
        cp_length = min(num_subcarriers, 288)
        guard_subcarriers = (0,0)
        # Define the resource grid.
        resource_grid = ResourceGrid(
            num_ofdm_symbols=14,
            fft_size=fft_size,
            subcarrier_spacing=subcarrier_spacing,
            num_tx=self.Num_tx,
            num_streams_per_tx=1,
            cyclic_prefix_length=cp_length,
            num_guard_carriers=guard_subcarriers,
            dc_null=False,
            pilot_pattern=self.Pilot_Pattern,
            dtype=tf.complex64
        )

        self.Resource_Grid_Mapper = ResourceGridMapper(resource_grid, dtype=tf.complex64)        
        
        self.AWGN = AWGN()

 
        self.Channel_Estimator = PUSCHLSChannelEstimator(
                        resource_grid,
                        pusch_config.dmrs.length,
                        pusch_config.dmrs.additional_position,
                        pusch_config.dmrs.num_cdm_groups_without_data,
                        interpolation_type='nn',
                        dtype=tf.complex64)

        rxtx_association = np.ones([self.Num_rx, self.Num_tx], bool)
        stream_management = StreamManagement(rxtx_association, pusch_config.num_layers)
        self.Mimo_Detector = LinearDetector("lmmse", "bit", "maxlog", resource_grid, stream_management,
                                    "qam", pusch_config.tb.num_bits_per_symbol, dtype=tf.complex64)
        

        self.Layer_Demapper = LayerDemapper(self.Layer_Mapper, num_bits_per_symbol=num_bits_per_symbol)
        self.TB_Decode = TBDecoder(self.TB_Encoder, output_dtype=tf.float32)

        self.tb_size = tb_size
        self.resource_grid = resource_grid
        self.pusch_config = pusch_config
        
    def update_pilots(self, pilots):
        self.Resource_Grid_Mapper._resource_grid.pilot_pattern.pilots = pilots
        """Channel Estimationand Detection will reflect this update since they reference the same object."""

    def sim(self, batch_size, channel_model, no_scaling, gen_prng_seq=None, return_tx_iq=False, return_channel=False):
        if gen_prng_seq:
            b = tf.reshape(tf.constant(generate_prng_seq(batch_size * self.Num_tx * self.tb_size, gen_prng_seq), dtype=tf.float32), [batch_size, self.Num_tx, self.tb_size])
        else:
            b = self.Binary_Source([batch_size, self.Num_tx, self.tb_size])

        c = self.TB_Encoder(b)
        x_map = self.Constellation_Mapper(c)
        x_layer = self.Layer_Mapper(x_map)
        x = self.Resource_Grid_Mapper(x_layer)

        y, h = channel_model(x)
        no = no_scaling * tf.math.reduce_variance(y)

        y = self.AWGN([y, no])

        if return_channel:
            if return_tx_iq:
                return b, c, y, x, h
            return b, c, y, h
        
        if return_tx_iq:
            return b, c, y, x
        return b, c, y
        
    
    def rec(self, y, no_ = 1e-3):
        h_hat, err_var = self.Channel_Estimator([y, no_])
        llr_det = self.Mimo_Detector([y, h_hat, err_var, no_])
        llr_layer = self.Layer_Demapper(llr_det)
        b_hat, tb_crc_status = self.TB_Decode(llr_layer)

        return h_hat, llr_det, b_hat, tb_crc_status
    
    def per(self, y, h, no):
        no_ = no
        h_hat, err_var = h, 0.
        llr_det = self.Mimo_Detector([y, h_hat, err_var, no_])
        llr_layer = self.Layer_Demapper(llr_det)
        b_hat, tb_crc_status = self.TB_Decode(llr_layer)

        return h_hat, llr_det, b_hat, tb_crc_status





# class ResidualBlock(tf.keras.Model):
#     r"""
#     This Keras layer implements a convolutional residual block made of two convolutional layers with ReLU activation, layer normalization, and a skip connection.
#     The number of convolutional channels of the input must match the number of kernel of the convolutional layers ``num_conv_channel`` for the skip connection to work.

#     Input
#     ------
#     : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
#         Input of the layer

#     Output
#     -------
#     : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
#         Output of the layer
#     """

#     def build(self, input_shape):

#         # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
#         self._layer_norm_1 = LayerNormalization(axis=(-1, -2, -3))
#         self._conv_1 = SeparableConv2D(filters= 64,
#                               kernel_size=[3,3],
#                               padding='same',
#                               activation=None)
#         # Layer normalization is done over the last three dimensions: time, frequency, conv 'channels'
#         self._layer_norm_2 = LayerNormalization(axis=(-1, -2, -3))
#         self._conv_2 = SeparableConv2D(filters= 128,
#                               kernel_size=[3,3],
#                               padding='same',
#                               activation=None)

#     def call(self, inputs):
#         z = self._layer_norm_1(inputs)
#         z = relu(z)
#         z = self._conv_1(z)
#         z = self._layer_norm_2(z)
#         z = relu(z)
#         z = self._conv_2(z) # [batch size, num time samples, num subcarriers, num_channels]
#         # Skip connection
#         z = z + inputs

#         return z

# class CustomNeuralReceiver(tf.keras.Model):
#     r"""
#     Keras layer implementing a residual convolutional neural receiver.

#     This neural receiver is fed with the post-DFT received samples, forming a resource grid of size num_of_symbols x fft_size, and computes LLRs on the transmitted coded bits.
#     These LLRs can then be fed to an outer decoder to reconstruct the information bits.

#     Input
#     ------
#     y_no: [batch size, num ofdm symbols, num subcarriers, 2*num rx antenna + 1], tf.float32
#         Concatenated received samples and noise variance.
# (
#     y : [batch size, num rx antenna, num ofdm symbols, num subcarriers], tf.complex
#         Received post-DFT samples.

#     no : [batch size], tf.float32
#         Noise variance. At training, a different noise variance value is sampled for each batch example.
# )
#     Output
#     -------
#     : [batch size, num ofdm symbols, num subcarriers, num_bits_per_symbol]
#         LLRs on the transmitted bits.
#     """

#     def __init__(self, training = False):
#         super(CustomNeuralReceiver, self).__init__()
#         self._training = training

#     def build(self, input_shape):

#         # Input convolution
#         self._input_conv = Conv2D(filters= 128,
#                                   kernel_size=[3,3],
#                                   padding='same',
#                                   activation=None)
#         # Residual blocks
#         self._res_block_1 = ResidualBlock()
#         self._res_block_2 = ResidualBlock()
#         self._res_block_3 = ResidualBlock()
#         self._res_block_4 = ResidualBlock()
#         # Output conv
#         self._output_conv = Conv2D(filters= 2,    # QPSK
#                                    kernel_size=[3,3],
#                                    padding='same',
#                                    activation=None)
        

#     @tf.function(jit_compile=True)
#     def call(self, inputs):
#         # Input conv
#         z = self._input_conv(inputs)
#         # Residual blocks
#         z = self._res_block_1(z)
#         z = self._res_block_2(z)
#         z = self._res_block_3(z)
#         z = self._res_block_4(z)
#         # Output conv
#         z = self._output_conv(z)
#         # if self._training == False:
#         #     z = tf.cast(z * (2**7), tf.int8)
#         return z
    

def load_weights(model, pretrained_weights_path):
    # Build Model with random input
    # Load weights
    with open(pretrained_weights_path, 'rb') as f:
        weights = pickle.load(f)
        model.set_weights(weights)
        print(f"Loaded pretrained weights from {pretrained_weights_path}")




import re
import math

def bitmask_to_indices(bitmask):
    indices = []
    index = 0
    while bitmask:
        if bitmask & 1:
            indices.append(index)
        bitmask >>= 1
        index += 1
    return indices

def config_parser(config_path):
    caseInfo = {}
    sysInfo = {}
    ue = {}
    chcfg = {}
    auxInfo = {}
    with open(config_path, 'r') as file:
        for num, line in enumerate(file, 1):
            line = line.strip()
            if line and not line.startswith('%'):  # Ignore empty or comment lines
                #read case information and store it in caseInfo
                key, value = line.split('=')
                key = key.strip()
                value = value.strip('; ').strip()
                if value.lower() == 'true': 
                    value = True
                elif value.lower() == 'false':
                    value = False
                elif value.isdigit():  # Convert to integer if the value is a number
                    value = int(value)
                if num < 3:
                    caseInfo[key] = value    
                else:
                    #read cell information  
                    if key.startswith('sys'):
                        _, value2 = key.split('.')
                        sysInfo[value2] = value
                    #read chcfg information
                    elif key.startswith('chcfg'): 
                        _, value2 = key.split('.')
                        chcfg[value2] = value
                    #read ue config
                    elif key.startswith('ue'): 
                        key2, value2 = key.split('.')
                        ue_idx = re.search(r"\{([^}]+)\}", key2)
                        ue_idx = ue_idx.group(1)
                        if ue_idx.isdigit():
                            ue_idx = int(ue_idx)
                        ue_idx = ue_idx - 1
                        if is_empty(ue, ue_idx) == 0 :
                            #create an empty config dictionary for ue_idx                     
                            ue[ue_idx] = {}
                        if value2 == 'rvIdx':
                            continue
                        if value2 == 'DmrsPortSetIdx':
                            ue[ue_idx][value2] = bitmask_to_indices(value)
                        else:
                            ue[ue_idx][value2] = value
                    else:
                        auxInfo[key] = value                
    return caseInfo, sysInfo, ue, chcfg, auxInfo    

def is_empty(dictionary, key):
    # Check if the key exists and if the value is considered "empty"
    if key in dictionary:
        return True
    return False

def fft_size_return(n):
    if n <= 1:
        return 1    
    if n >= 0.85*2**math.ceil(math.log2(n)):
        return 2**(math.ceil(math.log2(n))+1)
    else:
        return 2 ** math.ceil(math.log2(n))
    


PuschRecord = namedtuple("PuschRecord", [ "nPhyCellId",
    "nSFN", "nSlot", "nPDU", "nGroup", "nUlsch", "nUlcch", "nRachPresent",
    "nRNTI", "nUEId", "nBWPSize", "nBWPStart", "nSubcSpacing", "nCpType", "nULType",
    "nMcsTable", "nMCS", "nTransPrecode", "nTransmissionScheme", "nNrOfLayers",
    "nPortIndex", "nNid", "nSCID", "nNIDnSCID", "nNrOfAntennaPorts",
    "nVRBtoPRB", "nPMI", "nStartSymbolIndex", "nNrOfSymbols", "nResourceAllocType",
    "nRBStart", "nRBSize", "nTBSize", "nRV", "nHARQID", "nNDI", "nMappingType",
    "nDMRSTypeAPos", "nDMRSConfigType", "nNrOfCDMs", "nNrOfDMRSSymbols", "nDMRSAddPos",
    "nPTRSPresent", "nAck", "nAlphaScaling", "nBetaOffsetACKIndex", "nCsiPart1",
    "nBetaOffsetCsiPart1Index", "nCsiPart2", "nBetaOffsetCsiPart2Index",
    "nTpPi2BPSK", "nTPPuschID", "nRxRUIdx", "nUE", "nPduIdx",

    # New fields for channel and filenames
    "Channel_model", "Speed", "Delay_spread", "Esno_db",
    "Data_filename","Data_dirname"
    ]
)

def save_pickle(data, parent_name, group_name):
    """Saves data to a pickle file."""
    def save_to_pickle(data, filename):
        with open(filename, "wb") as f:
            pickle.dump(data, f)
    b, c, y, r = data
    save_to_pickle(b.numpy(), f'{parent_name}/{group_name}.b.pkl')
    save_to_pickle(c.numpy(), f'{parent_name}/{group_name}.c.pkl')
    save_to_pickle(y.numpy(), f'{parent_name}/{group_name}.y.pkl')
    save_to_pickle(r.numpy(), f'{parent_name}/{group_name}.r.pkl')

def save_hdf5(data, parent_name, group_name):
    b, c, y, r = data
    with h5py.File(f"{parent_name}.hdf5", "a") as hf:
        hf.create_dataset(f"{group_name}_b", data=b.numpy())
        hf.create_dataset(f"{group_name}_c", data=c.numpy())
        hf.create_dataset(f"{group_name}_y", data=y.numpy())
        hf.create_dataset(f"{group_name}_r", data=r.numpy())


def load_hdf5(parent_name, group_name):
    with h5py.File(f'{parent_name}.hdf5', "r") as f:
        b = f[f"{group_name}_b"][:]
        c = f[f"{group_name}_c"][:]
        y = f[f"{group_name}_y"][:]
        r = f[f"{group_name}_r"][:]
    return b, c, y, r

def load_pickle(parent_name, group_name):
    """Saves data to a pickle file."""
    def load_from_pickle(filename):
        with open(filename, "rb") as f:
            return pickle.load(f)

    b = load_from_pickle(f'{parent_name}/{group_name}.b.pkl')
    c = load_from_pickle(f'{parent_name}/{group_name}.c.pkl')
    y = load_from_pickle(f'{parent_name}/{group_name}.y.pkl')
    r = load_from_pickle(f'{parent_name}/{group_name}.r.pkl')

    return b, c, y, r


class ResidualBlock(tf.keras.Model):
    r"""
    This Keras layer implements a convolutional residual block made of two convolutional layers with ReLU activation, layer normalization, and a skip connection.
    The number of convolutional channels of the input must match the number of kernel of the convolutional layers ``num_conv_channel`` for the skip connection to work.

    Input
    ------
    : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
    Input of the layer

    Output
    -------
    : [batch size, num time samples, num subcarriers, num_conv_channel], tf.float
    Output of the layer
    """

    def build(self, input_shape):
        self._layer_norm_1 = LayerNormalization(axis=[-1,-2,-3])
        self._conv_1 = Conv2D(filters= 128,
            kernel_size=[3,3],
            padding='same',
            activation=None)

        self._layer_norm_2 = LayerNormalization(axis=[-1,-2,-3])
        self._conv_2 = Conv2D(filters= 128,
            kernel_size=[3,3],
            padding='same',
            activation=None)

    def call(self, inputs):
        z = self._layer_norm_1(inputs)
        z = relu(z)
        z = self._conv_1(z)
        z = self._layer_norm_2(z)
        z = relu(z)
        z = self._conv_2(z) # [batch size, num time samples, num subcarriers, num_channels]
        # Skip connection
        z = z + inputs

        return z

class CustomNeuralReceiver(tf.keras.Model):
    r"""
    Keras layer implementing a residual convolutional neural receiver.

    This neural receiver is fed with the post-DFT received samples, forming a resource grid of size num_of_symbols x fft_size, and computes LLRs on the transmitted coded bits.
    These LLRs can then be fed to an outer decoder to reconstruct the information bits.

    Input
    ------
    y_no: [batch size, num ofdm symbols, num subcarriers, 2*num rx antenna + 1], tf.float32
    Concatenated received samples and noise variance.
    (
    y : [batch size, num rx antenna, num ofdm symbols, num subcarriers], tf.complex
    Received post-DFT samples.

    no : [batch size], tf.float32
    Noise variance. At training, a different noise variance value is sampled for each batch example.
    )
    Output
    -------
    : [batch size, num ofdm symbols, num subcarriers, num_bits_per_symbol]
    LLRs on the transmitted bits.
    """

    def __init__(self, training = False):
        super(CustomNeuralReceiver, self).__init__()
        self._training = training

    def build(self, input_shape):

        # Input convolution
        self._input_conv = Conv2D(filters= 128,
        kernel_size=[3,3],
        padding='same',
        activation=None)
        # Residual blocks
        self._res_block_1 = ResidualBlock()
        self._res_block_2 = ResidualBlock()
        self._res_block_3 = ResidualBlock()
        self._res_block_4 = ResidualBlock()
        # Output conv
        self._output_conv = Conv2D(filters= 2, # QPSK
        kernel_size=[3,3],
        padding='same',
        activation=None)

    def call(self, inputs):
        # Input conv
        if self._training == False:
            padding_size = (-inputs.shape[1] % 48)
            if(padding_size != 0):
                padded_input_size = inputs.shape[1] + padding_size
                inputs = tf.concat([inputs, inputs[:,:padding_size,]],axis=1)
                inputs = tf.reshape(inputs, [-1,48,14,16])
        z = tf.linalg.l2_normalize(inputs, axis=[-1,-2,-3])
        z = self._input_conv(z)
        # Residual blocks
        z = self._res_block_1(z)
        z = self._res_block_2(z)
        z = self._res_block_3(z)
        z = self._res_block_4(z)
        # Output conv
        z = self._output_conv(z)

        if self._training == False:
            if padding_size != 0:
                z = tf.reshape(z, [-1,padded_input_size,14,2])
                z = z[:,:-(padding_size),]


        z = tf.concat([z[...,0:3,:],z[...,4:11,:], z[...,12:14,:]],axis=-2)
        z = tf.transpose(z, perm=[0,2,1,3])
        z = tf.reshape(z, [z.shape[0],(z.shape[1]*z.shape[2]*z.shape[3])])
        return z