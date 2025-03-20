from dataclasses import dataclass, field
from typing import List
from sionna.nr.utils import generate_prng_seq
from sionna.nr import PUSCHConfig, CarrierConfig, PUSCHDMRSConfig, TBConfig, PUSCHPilotPattern, TBEncoder,LayerMapper, LayerDemapper, TBDecoder, PUSCHLSChannelEstimator
from sionna.channel import AWGN, OFDMChannel, gen_single_sector_topology as gen_topology
from sionna.ofdm import LinearDetector, ResourceGrid, ResourceGridMapper, LMMSEInterpolator, MaximumLikelihoodDetector, KBestDetector
from sionna.mimo import StreamManagement
from sionna.mapping import Mapper
from sionna.utils import BinarySource
from sionna.channel.tr38901 import Antenna, AntennaArray, UMi, UMa, RMa, TDL, CDL
from tqdm import tqdm
import random

import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import tensorflow as tf
import numpy as np
import pickle
import h5py
import os
import time
import struct
import re
import math

from tensorflow.keras.layers import Layer, Conv2D, LayerNormalization, SeparableConv2D
from tensorflow.nn import relu

from collections import namedtuple

# sionna.config.xla_compat=True

CARRIER_FREQUENCY = 2.55e9
BANDWIDTH = 60
NUM_RX = 1
NUM_TX = 1
NUM_RX_ANT = 8
NUM_TX_ANT = 1
NUM_STREAMS_PER_TX = 1


Ue_Antenna = Antenna(polarization="single",
                polarization_type="V",
                antenna_pattern="38.901",
                carrier_frequency=CARRIER_FREQUENCY)

Gnb_AntennaArray = AntennaArray(num_rows=1,
                        num_cols=NUM_RX_ANT//2,
                        polarization="dual",
                        polarization_type="cross",
                        antenna_pattern="38.901",
                        carrier_frequency=CARRIER_FREQUENCY)


@dataclass
class SystemConfig:
    NCellId: int = 246
    FrequencyRange: int = 1
    BandWidth: int = BANDWIDTH
    Numerology: int = 1
    CpType: int = 0
    NTxAnt: int = NUM_TX_ANT
    NRxAnt: int = NUM_RX_ANT
    BwpNRb: int = 162
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
    Carrier_frequency: float = CARRIER_FREQUENCY  # Carrier frequency in Hz

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
                n_id=[My_Config.Ue[0].NnScIdId,My_Config.Ue[0].NnScIdId],
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
        self.dmrs._n_id = [value, value]

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
            num_tx=NUM_TX,
            num_streams_per_tx=NUM_STREAMS_PER_TX,
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

        rxtx_association = np.ones([NUM_RX, NUM_TX], bool)
        stream_management = StreamManagement(rxtx_association, pusch_config.num_layers)
        self.Mimo_Detector = LinearDetector("lmmse", "bit", "maxlog", resource_grid, stream_management,
                                    "qam", pusch_config.tb.num_bits_per_symbol, dtype=tf.complex64)
        
        self.Equalizer = LinearDetector("lmmse", "symbol", "maxlog", resource_grid, stream_management,
                                    "qam", pusch_config.tb.num_bits_per_symbol, dtype=tf.complex64)

        self.Layer_Demapper = LayerDemapper(self.Layer_Mapper, num_bits_per_symbol=num_bits_per_symbol)
        self.TB_Decoder = TBDecoder(self.TB_Encoder, output_dtype=tf.float32)

        self.tb_size = tb_size
        self.resource_grid = resource_grid
        self.pusch_config = pusch_config
        
    def update_pilots(self, pilots):
        self.Resource_Grid_Mapper._resource_grid.pilot_pattern.pilots = pilots
        """Channel Estimationand Detection will reflect this update since they reference the same object."""

    @tf.function()
    def sim(self, batch_size, channel_model, no_scaling, gen_prng_seq=None, return_tx_iq=False, return_channel=False):
        if gen_prng_seq:
            b = tf.reshape(tf.constant(generate_prng_seq(batch_size * NUM_TX * self.tb_size, gen_prng_seq), dtype=tf.float32), [batch_size, NUM_TX, self.tb_size])
        else:
            b = self.Binary_Source([batch_size, NUM_TX, self.tb_size])

        c = self.TB_Encoder(b)
        x_map = self.Constellation_Mapper(c)
        x_layer = self.Layer_Mapper(x_map)
        x = self.Resource_Grid_Mapper(x_layer)

        y, h = channel_model(x)
        no = no_scaling * tf.math.reduce_variance(y, axis=[-1,-2,-3,-4])

        y = self.AWGN([y, no])

        if return_channel:
            if return_tx_iq:
                return b, c, y, x, h
            return b, c, y, h
        
        if return_tx_iq:
            return b, c, y, x
        return b, c, y
    
    def ref(self, batch_size, dtype=tf.complex64):
        return tf.repeat(tf.transpose(tf.constant(self.pusch_config.dmrs_grid, dtype=dtype), [0, 2, 1])[None, None], repeats=batch_size, axis=0)
    
    def rec(self, y, snr_ = 1e3):
        no_ = tf.math.reduce_variance(y, axis=[-1,-2,-3,-4])/(snr_ + 1)
        h_hat, err_var = self.Channel_Estimator([y, no_])
        x_hat = self.Equalizer([y, h_hat, err_var, no_])
        llr_det = self.Mimo_Detector([y, h_hat, err_var, no_])
        llr_layer = self.Layer_Demapper(llr_det)
        b_hat, tb_crc_status = self.TB_Decoder(llr_layer)

        return h_hat, x_hat, llr_det, b_hat, tb_crc_status
    # def build_per(self, cov_mat_time, cov_mat_freq, cov_mat_space=None, order='t-f'):

    #     self.LMMSE_Channel_Estimator = PUSCHLSChannelEstimator(
    #             self.Resource_Grid_Mapper.resource_grid,
    #             self.pusch_config.dmrs.length,
    #             self.pusch_config.dmrs.additional_position,
    #             self.pusch_config.dmrs.num_cdm_groups_without_data,
    #             interpolator=LMMSEInterpolator(
    #                 pilot_pattern=self.Resource_Grid_Mapper._resource_grid.pilot_pattern,
    #                 cov_mat_time=cov_mat_time,
    #                 cov_mat_freq=cov_mat_freq,
    #                 cov_mat_space=cov_mat_space,
    #                 order=order                            
    #                 ),
    #             dtype=tf.complex64)

    def per(self, y, h, no):       
        h_hat, err_var = h, 0.
        x_hat = self.Equalizer([y, h_hat, err_var, no])
        llr_det = self.Mimo_Detector([y, h_hat, err_var, no])
        llr_layer = self.Layer_Demapper(llr_det)
        b_hat, tb_crc_status = self.TB_Decoder(llr_layer)

        return h_hat, x_hat, llr_det, b_hat, tb_crc_status


def generate_data(name: str,
        data_dir: str,
        pusch_configs: List[MyPUSCHConfig],
        channel_scenarios: List[str],
        esno_dbs: List[float],
        slots: List[int],
        save_dataset: str = None):
    assert save_dataset in [None, 'hdf5', 'pickle']
    """set up save directory"""
    pusch_records=[]
    parquet_dir = f'{data_dir}/parquet'
    if save_dataset: os.makedirs(parquet_dir, exist_ok=True)
    hdf5_dir = f'{data_dir}/hdf5'
    if save_dataset == 'hdf5': os.makedirs(hdf5_dir, exist_ok=True)
    pickle_dir = f'{data_dir}/pickle'
    if save_dataset == 'pickle': os.makedirs(pickle_dir, exist_ok=True)

    """set up for per slot"""
    len_per_case = len(slots)

    total_iterations = len(pusch_configs) * len(esno_dbs) * len_per_case


    """generate ..."""
    with tqdm(total=total_iterations, desc="Generating Data") as pbar:
        for config_idx, pusch_config in enumerate(pusch_configs):
           
            Pusch_Pilots = {}
            Pusch_Slots = set(slots)
            for n, slot in enumerate(Pusch_Slots):
                pusch_config_i = pusch_config.clone()
                pusch_config_i.carrier.slot_number = slot
                pilot_pattern_i = PUSCHPilotPattern([pusch_config_i], dtype=tf.complex64)
                Pusch_Pilots[slot] = pilot_pattern_i.pilots
            len_pusch = len(Pusch_Slots)

            """set up for all case"""
            carrier_frequency = pusch_config.My_Config.Carrier_frequency

            simulator = MySimulator(pusch_config)

            channel_scenario = random.choice(channel_scenarios)
            """channel_scenario form: {CDL}-{A|B|C|D|E}-{delay_spread (ns)}-{speed (m/s)}
                                or {Umi|Uma}-{low|high}-[OnPL]-[OnSF]-{delay_spread (ns)}-{speed (m/s)}

                Ex: CDL-A-150-10"""

            chn_scn = channel_scenario.split('-')
            
            model = chn_scn[1]
            channel = chn_scn[0]
            enable_pl = True if 'OnPL' in chn_scn else False # Umi/Uma enable pathloss
            enable_sf = True if 'OnSF' in chn_scn else False # Umi/Uma enable shadow fading

            speed = float(chn_scn[-1])
            delay_spread = float(chn_scn[-2])
            

            if 'CDL' == channel:
                channel_model = CDL(model = model,
                                        delay_spread = delay_spread*1e-9,
                                        carrier_frequency = carrier_frequency,
                                        ut_array = Ue_Antenna,
                                        bs_array = Gnb_AntennaArray,
                                        direction = "uplink",
                                        min_speed = speed,
                                        max_speed = speed)

            else:
                if 'Umi' == channel:
                    channel_model = UMi(carrier_frequency = carrier_frequency,
                                        o2i_model = model,
                                        ut_array = Ue_Antenna,
                                        bs_array = Gnb_AntennaArray,
                                        direction = "uplink",
                                        enable_pathloss = enable_pl,
                                        enable_shadow_fading = enable_sf)
                elif 'Uma' == channel:
                    channel_model = UMa(carrier_frequency = carrier_frequency,
                                        o2i_model = model,
                                        ut_array = Ue_Antenna,
                                        bs_array = Gnb_AntennaArray,
                                        direction = "uplink",
                                        enable_pathloss = enable_pl,
                                        enable_shadow_fading = enable_sf)

            simulator = MySimulator(pusch_config)

            channel_i = OFDMChannel(channel_model=channel_model, resource_grid=simulator.resource_grid,
                                    add_awgn=False, normalize_channel=True, return_channel=True)
            
            for esno_db in esno_dbs:
                no_scaling = tf.cast(pow(10., -esno_db / 10.), tf.float32)
                if channel in ['Umi', 'Uma']:
                    channel_i._cir_sampler.set_topology(*gen_topology(1,1,channel.lower(),min_ut_velocity=speed, max_ut_velocity=speed))

                for n,slot in enumerate(slots):
                    status_str = f"(config {config_idx} | channel {channel_scenario} | {esno_db} dB | slot {slot} | Sample: {n+1}/{len_per_case}"
                    pbar.set_description(status_str)
                    simulator.pusch_config.carrier.frame_number = (n // len_pusch) % 1023
                    simulator.pusch_config.carrier.slot_number = slot
                    simulator.update_pilots(Pusch_Pilots[slot])
                    
                    b, c, y = simulator.sim(1, channel_i, no_scaling, return_channel=False)

                    assert b.shape[0] == b.shape[1] == c.shape[0] == c.shape[1] == y.shape[0] == y.shape[1] == 1
                    b = tf.cast(b, dtype=tf.uint8)[0][0]
                    c = tf.cast(c, dtype=tf.uint8)[0][0]
                    r = tf.transpose(tf.constant(simulator.pusch_config.dmrs_grid, dtype=y.dtype), [0, 2, 1])
                    y =  y[0][0]

                    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")


                    # print(y, r)
                    if save_dataset == 'hdf5':
                        save_hdf5([b,c,y,r], f'{hdf5_dir}/{name}', timestamp)
                    if save_dataset == 'pickle':
                        save_pickle([b,c,y,r], f'{pickle_dir}/{name}', timestamp)

                    
                    
                    pusch_records.append(PuschRecord(
                                        nPhyCellId=simulator.pusch_config.carrier.n_cell_id,
                                        nSFN=simulator.pusch_config.carrier.frame_number,
                                        nSlot=simulator.pusch_config.carrier.slot_number,
                                        nPDU=1,
                                        nGroup=1,
                                        nUlsch=1,
                                        nUlcch=0,
                                        nRachPresent=0,
                                        nRNTI=simulator.pusch_config.n_rnti,
                                        nUEId=0,
                                        nBWPSize=simulator.pusch_config.n_size_bwp,
                                        nBWPStart=simulator.pusch_config.n_start_bwp,
                                        nSubcSpacing=simulator.pusch_config.carrier.mu,
                                        nCpType=simulator.pusch_config.My_Config.Sys.CpType,
                                        nULType=0,
                                        nMcsTable=simulator.pusch_config.tb.mcs_table - 1,
                                        nMCS=simulator.pusch_config.tb.mcs_index,
                                        nTransPrecode=simulator.pusch_config.My_Config.Ue[0].TransformPrecoding,
                                        nTransmissionScheme=simulator.pusch_config.My_Config.Ue[0].CodeBookBased,
                                        nNrOfLayers=simulator.pusch_config.num_layers,
                                        nPortIndex=simulator.pusch_config.dmrs.dmrs_port_set,
                                        nNid=simulator.pusch_config.tb.n_id,
                                        nSCID=simulator.pusch_config.dmrs.n_scid,
                                        nNIDnSCID=simulator.pusch_config.dmrs.n_id[0],
                                        nNrOfAntennaPorts=simulator.pusch_config.My_Config.Sys.NRxAnt,
                                        nVRBtoPRB=0,
                                        nPMI=simulator.pusch_config.My_Config.Ue[0].Tpmi,
                                        nStartSymbolIndex=simulator.pusch_config.symbol_allocation[0],
                                        nNrOfSymbols=simulator.pusch_config.symbol_allocation[1],
                                        nResourceAllocType=1,
                                        nDMRSTypeAPos=simulator.pusch_config.dmrs.type_a_position,
                                        nRBStart=simulator.pusch_config.first_resource_block,
                                        nRBSize=simulator.pusch_config.num_resource_blocks,
                                        nTBSize=(simulator.pusch_config.tb_size//8),
                                        nRV=simulator.pusch_config.My_Config.Sys.rvSeq,
                                        nHARQID=n % 16,
                                        nNDI=1,
                                        nMappingType=simulator.pusch_config.My_Config.Ue[0].PuschMappingType,
                                        nDMRSConfigType=simulator.pusch_config.My_Config.Ue[0].DmrsConfigurationType,
                                        nNrOfCDMs=simulator.pusch_config.dmrs.num_cdm_groups_without_data,
                                        nNrOfDMRSSymbols=simulator.pusch_config.dmrs.length,
                                        nDMRSAddPos=simulator.pusch_config.dmrs.additional_position,
                                        nPTRSPresent=simulator.pusch_config.My_Config.Ue[0].Ptrs,
                                        nAck=simulator.pusch_config.My_Config.Ue[0].OAck,
                                        nAlphaScaling=simulator.pusch_config.My_Config.Ue[0].ScalingFactor,
                                        nBetaOffsetACKIndex=simulator.pusch_config.My_Config.Ue[0].IHarqAckOffset,
                                        nCsiPart1=simulator.pusch_config.My_Config.Ue[0].OCsi1,
                                        nBetaOffsetCsiPart1Index=simulator.pusch_config.My_Config.Ue[0].ICsi1Offset,
                                        nCsiPart2=simulator.pusch_config.My_Config.Ue[0].OCsi2,
                                        nBetaOffsetCsiPart2Index=simulator.pusch_config.My_Config.Ue[0].ICsi2Offset,
                                        nTpPi2BPSK=simulator.pusch_config.My_Config.Ue[0].TpPi2Bpsk,
                                        nTPPuschID=simulator.pusch_config.My_Config.Ue[0].NRsId,
                                        nRxRUIdx=np.arange(0, simulator.pusch_config.My_Config.Sys.NRxAnt),
                                        nUE=1,
                                        nPduIdx=[0],
                                        Channel_model=f"{channel}-{model}",
                                        Speed=speed,
                                        Delay_spread=delay_spread,
                                        Esno_db=esno_db,
                                        Data_filename=timestamp,
                                        Data_dirname=name
                                    )
                                )
                    pbar.update(1)  # Increment progress

            df = pd.DataFrame.from_records(pusch_records, columns=PuschRecord._fields)
            if save_dataset: df.to_parquet(f'{parquet_dir}/{name}.parquet', engine="pyarrow")
    return df



def load_weights(model, pretrained_weights_path):
    # Build Model with random input
    # Load weights
    with open(pretrained_weights_path, 'rb') as f:
        weights = pickle.load(f)
        model.set_weights(weights)
        print(f"Loaded pretrained weights from {pretrained_weights_path}")



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
      

        z = inputs
        z = self._input_conv(z)
        # Residual blocks
        z = self._res_block_1(z)
        z = self._res_block_2(z)
        z = self._res_block_3(z)
        z = self._res_block_4(z)
        # Output conv
        z = self._output_conv(z)

        return z





def setCfgReq(puschCfg: MyPUSCHConfig, slots, dir):
    bandwidth = puschCfg.My_Config.Sys.BandWidth
    fft_size = 4096
    filename = os.path.join(dir, 'cfgReq.cfg')
    
    slot_config_template = "1,1,1,1,1,1,1,1,1,1,1,1,1,1"  # 14 ones
    default_config_template = "2,2,2,2,2,2,2,2,2,2,2,2,2,2"  # 14 twos
    
    xml_content = """<?xml version="1.0"?>

<TestConfig>
	<numSlots>20</numSlots>
"""
    for i in range(NUM_RX_ANT):
        xml_content += f"\t<uliq_car0_ant{i}>rx_ant_{i}.bin</uliq_car0_ant{i}>\n"
    
    xml_content += f"""	<ul_ref_out>ref.txt</ul_ref_out>
	<start_frame_number>0</start_frame_number>
	<start_slot_number>0</start_slot_number>
</TestConfig>

<ConfigReq>
	<nCarrierIdx>0</nCarrierIdx>
	<nDMRSTypeAPos>3</nDMRSTypeAPos>
	<nPhyCellId>{puschCfg.carrier.n_cell_id}</nPhyCellId>
	<nDLBandwidth>{bandwidth}</nDLBandwidth>
	<nULBandwidth>{bandwidth}</nULBandwidth>
	<nDLFftSize>{fft_size}</nDLFftSize>
	<nULFftSize>{fft_size}</nULFftSize>
	<nNrOfTxAnt>NUM_RX_ANT</nNrOfTxAnt>
	<nNrOfRxAnt>NUM_RX_ANT</nNrOfRxAnt>
	<nCarrierAggregationLevel>0</nCarrierAggregationLevel>
	<nFrameDuplexType>1</nFrameDuplexType>
	<nSubcCommon>1</nSubcCommon>
	<nTddPeriod>20</nTddPeriod>
"""
    
    for i in range(20):
        config_value = slot_config_template if i in slots else default_config_template
        xml_content += f"\t<sSlotConfig{i}>{config_value}</sSlotConfig{i}>\n"
    
    xml_content += "\t<nCyclicPrefix>0</nCyclicPrefix>\n</ConfigReq>\n\n<RxConfig>\n"
    
    for i in range(20):
        slot_value = f"slot{i}.cfg" if i in slots else "null.cfg"
        xml_content += f"\t<SlotNum{i}>{slot_value}</SlotNum{i}>\n"
    
    xml_content += """</RxConfig>
"""
    
    with open(filename, 'w') as f:
        f.write(xml_content)
    
    print(f"Config Request saved to {filename}")

def setUlCfgReq(puschCfg: MyPUSCHConfig, harqIdx, dir):

    filename = os.path.join(dir, f'slot{puschCfg.carrier.slot_number}.cfg')

    # Define the XML structure dynamically
    xml_content = f"""<?xml version="1.0"?>

<Ul_Config_Req>

	<UlConfigReqL1L2Header>
		<nSFN>{puschCfg.carrier.frame_number}</nSFN>
		<nSlot>{puschCfg.carrier.slot_number}</nSlot>
		<nPDU>{1}</nPDU>
		<nGroup>{1}</nGroup>
		<nUlsch>{1}</nUlsch>
		<nUlcch>{0}</nUlcch>
		<nRachPresent>{0}</nRachPresent>
	</UlConfigReqL1L2Header>

	<UL_SCH_PDU0>
		<nRNTI>{puschCfg.n_rnti}</nRNTI>
		<nUEId>{0}</nUEId>
		<nBWPSize>{puschCfg.n_size_bwp}</nBWPSize>
		<nBWPStart>{puschCfg.n_start_bwp}</nBWPStart>
		<nSubcSpacing>{puschCfg.carrier.mu}</nSubcSpacing>
		<nCpType>{puschCfg.My_Config.Sys.CpType}</nCpType>
		<nULType>{0}</nULType>
		<nMcsTable>{puschCfg.tb.mcs_table - 1}</nMcsTable>
		<nMCS>{puschCfg.tb.mcs_index}</nMCS>
		<nTransPrecode>{puschCfg.My_Config.Ue[0].TransformPrecoding}</nTransPrecode>
		<nTransmissionScheme>{puschCfg.My_Config.Ue[0].CodeBookBased}</nTransmissionScheme>
		<nNrOfLayers>{puschCfg.num_layers}</nNrOfLayers>
		<nPortIndex0>{puschCfg.dmrs.dmrs_port_set[0]}</nPortIndex0>
		<nNid>{puschCfg.tb.n_id}</nNid>
		<nSCID>{puschCfg.dmrs.n_scid}</nSCID>
		<nNIDnSCID>{puschCfg.dmrs.n_id[0]}</nNIDnSCID>
		<nNrOfAntennaPorts>{puschCfg.My_Config.Sys.NRxAnt}</nNrOfAntennaPorts>
		<nVRBtoPRB>{0}</nVRBtoPRB>
		<nPMI>{puschCfg.My_Config.Ue[0].Tpmi}</nPMI>
		<nStartSymbolIndex>{puschCfg.symbol_allocation[0]}</nStartSymbolIndex>
		<nNrOfSymbols>{puschCfg.symbol_allocation[1]}</nNrOfSymbols>
		<nResourceAllocType>{1}</nResourceAllocType>
		<nRBStart>{puschCfg.first_resource_block}</nRBStart>
		<nRBSize>{puschCfg.num_resource_blocks}</nRBSize>
		<nTBSize>{(puschCfg.tb_size//8)}</nTBSize>
		<nRV>{puschCfg.My_Config.Sys.rvSeq}</nRV>
		<nHARQID>{harqIdx}</nHARQID>
		<nNDI>{1}</nNDI>
		<nMappingType>{puschCfg.My_Config.Ue[0].PuschMappingType}</nMappingType>
		<nDMRSConfigType>{puschCfg.My_Config.Ue[0].DmrsConfigurationType}</nDMRSConfigType>
		<nNrOfCDMs>{puschCfg.dmrs.num_cdm_groups_without_data}</nNrOfCDMs>
		<nNrOfDMRSSymbols>{puschCfg.dmrs.length}</nNrOfDMRSSymbols>
		<nDMRSAddPos>{puschCfg.dmrs.additional_position}</nDMRSAddPos>
		<nPTRSPresent>{puschCfg.My_Config.Ue[0].Ptrs}</nPTRSPresent>
		<nAck>{puschCfg.My_Config.Ue[0].OAck}</nAck>
		<nAlphaScaling>{puschCfg.My_Config.Ue[0].ScalingFactor}</nAlphaScaling>
		<nBetaOffsetACKIndex>{puschCfg.My_Config.Ue[0].IHarqAckOffset}</nBetaOffsetACKIndex>
		<nCsiPart1>{puschCfg.My_Config.Ue[0].OCsi1}</nCsiPart1>
		<nBetaOffsetCsiPart1Index>{puschCfg.My_Config.Ue[0].ICsi1Offset}</nBetaOffsetCsiPart1Index>
		<nCsiPart2>{puschCfg.My_Config.Ue[0].OCsi2}</nCsiPart2>
		<nBetaOffsetCsiPart2Index>{puschCfg.My_Config.Ue[0].ICsi2Offset}</nBetaOffsetCsiPart2Index>
		<nTpPi2BPSK>{puschCfg.My_Config.Ue[0].TpPi2Bpsk}</nTpPi2BPSK>
		<nTPPuschID>{puschCfg.My_Config.Ue[0].NRsId}</nTPPuschID>
		<nRxRUIdx0>{0}</nRxRUIdx0>
		<nRxRUIdx1>{1}</nRxRUIdx1>
		<nRxRUIdx2>{2}</nRxRUIdx2>
		<nRxRUIdx3>{3}</nRxRUIdx3>
		<nRxRUIdx4>{4}</nRxRUIdx4>
		<nRxRUIdx5>{5}</nRxRUIdx5>
		<nRxRUIdx6>{6}</nRxRUIdx6>
		<nRxRUIdx7>{7}</nRxRUIdx7>
	</UL_SCH_PDU0>

	<PUSCH_GROUP_INFO0>
		<nUE>{1}</nUE>
		<nPduIdx0>{0}</nPduIdx0>
	</PUSCH_GROUP_INFO0>

</Ul_Config_Req>
"""
    with open(filename, 'w') as file:
        file.write(xml_content)
    print(f"UL Config Request Slot {puschCfg.carrier.slot_number} saved to {filename}")

def set_null(dir):
    filename = os.path.join(dir, 'null.cfg')

    xml_content = f"""<?xml version="1.0"?>

<Ul_Config_Req>
	<UlConfigReqL1L2Header>
		<nSFN>0</nSFN>
		<nSlot>0</nSlot>
		<nPDU>0</nPDU>
		<nGroup>0</nGroup>
		<nUlsch>0</nUlsch>
		<nUlcch>0</nUlcch>
		<nRachPresent>0</nRachPresent>
	</UlConfigReqL1L2Header>
</Ul_Config_Req>
"""
    with open(filename, 'w') as file:
        file.write(xml_content)
    print(f"Null Config saved to {filename}")
    
def setRxData(rxSigFreq, dir):
    # rxSigFreq = tf.reshape(rxSigFreq, -1)
    rxSigFreq = tf.stack((tf.math.real(rxSigFreq), tf.math.imag(rxSigFreq)), axis=-1)
    
    rxSigFreq = rxSigFreq/tf.math.reduce_max(tf.abs(rxSigFreq))

    rxSigFreq = tf.cast(tf.round(rxSigFreq*2**13), tf.int16)
    rxSigFreq = tf.reshape(rxSigFreq,[NUM_RX_ANT,-1])
    
    for rxIdx in range(NUM_RX_ANT):
        file_path = os.path.join(dir, f'rx_ant_{rxIdx}.bin')
        with open(file_path, 'wb') as file:
            file.write(rxSigFreq[rxIdx])
    return rxSigFreq



def setRef(inBits, dir):
    filename = os.path.join(dir, 'ref.txt')
    with open(filename, 'w') as file:
        file.write(f'##----------------------------------------------------------------------------\n')
    return 
    
    # with open(filename, 'w') as file:
    #     file.write(f'##----------------------------------------------------------------------------\n')
    #     fn = -1
    #     for n,payload in enumerate(inBits):
    #         # payload = inBits[:, i]
    #         if n in [4,5,14,15]:
    #             fn = fn + 1
    #             tbSize = payload.shape[-1]//8
    #             file.write(f'#type[PUSCH] fn[{fn}] slot[{n}] sym[0] carrier[0] chanId[0] len[{tbSize}]\n')
    #             file.write(f'\t  #ta[0] cqi[0.0] stat[1]\n')
    #             file.write(f'\t  #data[\n')
    #             Q = tbSize//64
    #             payload = tf.math.reduce_sum(tf.reshape(payload,[-1, 8]) * tf.constant([[128, 64, 32, 16, 8, 4, 2, 1]], dtype=tf.uint8), axis=1)
    #             # print(payload.shape, Q, r)
    #             for q in range(Q):
    #                 file.write('\t        ')
    #                 for r in range(64):
    #                     file.write(f'{payload[64*q + r]:3d}, ')
    #                 file.write('\n')
    #             file.write('\t        ')
    #             for r in range(tbSize%64-1):
    #                 file.write(f'{payload[64*Q + r]:3d}, ')
    #             file.write(f'{payload[-1]:3d}\n')
    #             file.write(f'\t       ]\n')
    #             file.write(f'------------------------------------------------------------------------------\n')
    #         else: 
    #             file.write(f'##----------------------------------------------------------------------------\n')




"""Predict and loss function"""
def loss_cal(pred, labels):
  bce = tf.nn.sigmoid_cross_entropy_with_logits(labels, pred)
  bce = tf.reduce_mean(bce)
  loss = bce
  return loss

def compute_ber(b, b_hat):
    """Computes the bit error rate (BER) between two binary tensors.

    Input
    -----
        b : tf.float32
            A tensor of arbitrary shape filled with ones and
            zeros.

        b_hat : tf.float32
            A tensor of the same shape as ``b`` filled with
            ones and zeros.

    Output
    ------
        : tf.float64
            A scalar, the BER.
    """
    ber = tf.not_equal(b, b_hat)
    ber = tf.cast(ber, tf.float64) # tf.float64 to suport large batch-sizes
    return tf.reduce_mean(ber)

@tf.function(jit_compile=True)
def predict(model, y, r):
    assert len(y.shape) == len(y.shape)  == 5, "y,r shape should be [batch_size, num_tx/rx, num_antennas, num_ofdm_symbols, num_subcarriers]"
    assert y.shape[1] == r.shape[1] == 1, "num_tx/rx should be 1"
    
    def preproc(tensor):
        tensor = tensor[:,0]
        tensor = tf.transpose(tensor, [0, 3,2,1])
        tensor = tf.concat([tf.math.real(tensor), tf.math.imag(tensor)], axis=-1)
        return tensor
    y = preproc(y)
    y = (y - tf.math.reduce_mean(y, axis=[-1,-2,-3,-4], keepdims=True))/(tf.math.reduce_std(y, axis=[-1,-2,-3,-4], keepdims=True))

    r = preproc(r)

    inputs = tf.concat([y, r], axis=-1)
    
    padding_size = (-inputs.shape[1] % 48)
    padded_input_size = inputs.shape[1]
    if(padding_size != 0):
        padded_input_size = padded_input_size + padding_size
        inputs = tf.concat([inputs, inputs[:,:padding_size,]],axis=1)
    inputs = tf.reshape(inputs, [-1,48,14,18])


    preds = model(inputs)
    
    preds = tf.reshape(preds, [-1,padded_input_size,14,2])
    if padding_size != 0:
        preds = preds[:,:-(padding_size),]


    preds = tf.concat([preds[...,0:3,:],preds[...,4:11,:], preds[...,12:14,:]],axis=-2)
    preds = tf.transpose(preds, perm=[0,2,1,3])
    preds = tf.reshape(preds, [-1,(preds.shape[1]*preds.shape[2]*preds.shape[3])])
    
    return preds

def data_reader(file_path, shape=[8,14,-1]):
    """Reads complex int16 data from a binary file."""
    freq = []
    with open(file_path, 'rb') as file:
        binary_data = file.read()
        for i in range(0, len(binary_data), 4):
            real = binary_data[i:i+2]
            imag = binary_data[i+2:i+4]
            if len(real) == 2:
                real_part = struct.unpack('<h', real)[0]
                imag_part = struct.unpack('<h', imag)[0]
                freq.append(complex(real_part, imag_part))
    return np.array(freq, dtype=np.complex64).reshape(shape)  # Use np.complex64 for efficient storage (8 ant x 14 Sym x Num subcarrier)

def data_writer(file_path, data):
    """Writes complex int16 data into a binary file."""
    data = data.flatten()
    with open(file_path, 'wb') as file:
        for value in data:
            real_part = struct.pack('<h', int(value.real))
            imag_part = struct.pack('<h', int(value.imag))
            file.write(real_part + imag_part)



def pusch_config_from_pd_row(pd_row: pd.Series) -> MyConfig:
    sysCfg = SystemConfig(
        NCellId=int(pd_row.nPhyCellId),
        CpType=int(pd_row.nCpType),
        BwpNRb=int(pd_row.nBWPSize),
        BwpRbOffset=int(pd_row.nBWPStart)
    )
    ueCfg = UeConfig(
        TransformPrecoding=int(pd_row.nTransPrecode),
        Rnti=int(pd_row.nRNTI),
        nId=int(pd_row.nNid),
        NLayers=int(pd_row.nNrOfLayers),
        FirstSymb=int(pd_row.nStartSymbolIndex),
        NPuschSymbAll=int(pd_row.nNrOfSymbols),
        FirstPrb=int(pd_row.nRBStart),
        NPrb=int(pd_row.nRBSize),
        McsTable=int(pd_row.nMcsTable),
        Mcs=int(pd_row.nMCS),
        nScId=int(pd_row.nSCID),
        NnScIdId=int(pd_row.nNIDnSCID),
        DmrsConfigurationType=int(pd_row.nDMRSConfigType),
        DmrsDuration=int(pd_row.nNrOfDMRSSymbols),
        DmrsAdditionalPosition=int(pd_row.nDMRSAddPos),
        PuschMappingType=int(pd_row.nMappingType),
        DmrsTypeAPosition=int(pd_row.nDMRSTypeAPos),
        Ptrs=int(pd_row.nPTRSPresent),
        OAck=int(pd_row.nAck),
        OCsi1=int(pd_row.nCsiPart1),
        OCsi2=int(pd_row.nCsiPart2),
        TpPi2Bpsk=int(pd_row.nTpPi2BPSK)
    )
    myCfg = MyConfig(sysCfg, [ueCfg])
    puschCfg = MyPUSCHConfig(myCfg)
    return puschCfg

def get_unique_configs(df: pd.DataFrame):
    Pusch_used_Cols = ['nPhyCellId', 'nCpType', 'nSubcSpacing', 'nBWPSize', 'nBWPStart', 'nSlot',
    'nDMRSConfigType', 'nNrOfDMRSSymbols', 'nDMRSAddPos', 'nPortIndex', 'nNIDnSCID', 'nSCID', 'nNrOfCDMs', 'nDMRSTypeAPos',
    'nNid', 'nMcsTable', 'nMCS',
    'nMappingType', 'nNrOfLayers', 'nTransmissionScheme', 'nPMI', 'nTransPrecode', 'nRNTI',
    'nStartSymbolIndex', 'nNrOfSymbols', 'nRBStart', 'nRBSize',
    'nPTRSPresent', 'nAck', 'nCsiPart1', 'nCsiPart2', 'nTpPi2BPSK']

    df_copy = df[Pusch_used_Cols].copy()
    for col in df_copy.columns:
        if isinstance(df_copy[col].iloc[0], np.ndarray):  # Check first element type
            df_copy[col] = df_copy[col].apply(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)

    return df_copy.reset_index().groupby(list(df_copy.columns)).agg(indices=('index',list)).reset_index()


def data_loader(df, dir, saved_dataset='hdf5'):
    assert saved_dataset in ['hdf5', 'pickle'], "saved data set should be 'pickle' or 'hdf5'."
    assert 'index' in df.columns, "DataFrame must contain a column named 'index'. Reading parquet should be 'df = pd.read_parquet(...).reset_index()'"
    for pusch_record in df.itertuples():
        data_filename = pusch_record.Data_filename
        data_dirname = pusch_record.Data_dirname
        esno_db = pusch_record.Esno_db
        index = pusch_record.index
        if saved_dataset == 'hdf5':
            _,c,y, r = load_hdf5(f'{dir}/{data_dirname}', data_filename)
        else:
            _,c,y, r = load_pickle(f'{dir}/{data_dirname}', data_filename)
        yield index, esno_db, c, y, r

def preprocessing(index, esno_db, c, y, r):
    y = tf.concat([tf.math.real(y), tf.math.imag(y)], axis = 0)
    y = tf.transpose(y, perm=[2,1,0])
    y = (y - tf.reduce_mean(y)) / tf.math.reduce_std(y)
    r = tf.concat([tf.math.real(r), tf.math.imag(r)], axis = 0)
    r = tf.transpose(r, perm=[2,1,0])
    return index, esno_db, c, y, r

def poly_hash(arr, base=31, mod=1_000_000_007):
    arr = np.reshape(arr, [-1, arr.shape[-1]])
    hash_val = [0]*arr.shape[0]
    for n in range(arr.shape[0]):
        for num in arr[n]:
            hash_val[n] = (hash_val[n] * base + num) % mod
    return hash_val