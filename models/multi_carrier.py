from models import vcm

def get_sub_carriers(B_Hz):
    # Determine number of sub-carriers
    if B_Hz <= 100e6:
        return 1
    if B_Hz <= 200e6:
        return 2
    if B_Hz <= 300e6:
        return 3

def compute_passes_throughput(tof_s_list, fspl_dB_list, Ptx_dBm_array, Gtx_dBi, GT_dBK, B_Hz,
                              alpha, EsN0_req_dB_array, eta_bitsym_2darray, min_margin_dB):

    carriers = get_sub_carriers(B_Hz)

    eta_bitsym_array = eta_bitsym_2darray[:, carriers - 1]

    return vcm.compute_passes_throughput(tof_s_list, fspl_dB_list, Ptx_dBm_array, Gtx_dBi, GT_dBK, B_Hz,
                                         alpha, EsN0_req_dB_array, eta_bitsym_array, min_margin_dB)