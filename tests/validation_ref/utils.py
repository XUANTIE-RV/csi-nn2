import os
import ctypes
import subprocess


MAX_DIM = 8


class csinn_quant_info(ctypes.Structure):
    _fields_ = [
        ("zero_point", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("multiplier", ctypes.c_int32),
        ("shift", ctypes.c_int32),
        ("min", ctypes.c_float),
        ("max", ctypes.c_float),
    ]


class csinn_model(ctypes.Structure):
    _fields_ = [
        ("bm_path", ctypes.c_char_p),
        ("bm_addr", ctypes.c_void_p),
        ("bm_size", ctypes.c_size_t),
        ("save_mode", ctypes.c_int32),
        ("priority", ctypes.c_int32),
    ]


class csinn_tensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("dtype", ctypes.c_int32),
        ("mtype", ctypes.c_int32),
        ("dim", ctypes.c_int32 * MAX_DIM),
        ("dim_count", ctypes.c_int32),
        ("is_const", ctypes.c_uint32),
        ("name", ctypes.c_char_p),
        ("layout", ctypes.c_int32),
        ("quant_channel", ctypes.c_int32),
        ("qinfo", ctypes.POINTER(csinn_quant_info)),
        ("sess", ctypes.c_void_p),
    ]


class csinn_session(ctypes.Structure):
    _fields_ = [
        ("base_dtype", ctypes.c_int32),
        ("base_layout", ctypes.c_int32),
        ("base_api", ctypes.c_int32),
        ("base_run_mode", ctypes.c_int32),
        ("base_quant_type", ctypes.c_int32),
        ("model", csinn_model),
        ("debug_level", ctypes.c_int32),
        ("profiler_level", ctypes.c_int32),
        ("input_num", ctypes.c_int32),
        ("output_num", ctypes.c_int32),
        ("input", ctypes.POINTER(ctypes.POINTER(csinn_tensor))),
        ("output", ctypes.POINTER(ctypes.POINTER(csinn_tensor))),
        ("td", ctypes.c_void_p),
        ("dynamic_shape", ctypes.c_bool),
    ]


class csinn_callback(ctypes.Structure):
    _fields_ = [
        ("init", ctypes.CFUNCTYPE(ctypes.c_int)),
        ("est", ctypes.CFUNCTYPE(ctypes.c_int)),
        ("exec", ctypes.CFUNCTYPE(ctypes.c_int)),
        ("caps", ctypes.CFUNCTYPE(ctypes.c_int)),
        ("perf", ctypes.CFUNCTYPE(ctypes.c_int)),
    ]


class csinn_params_base(ctypes.Structure):
    _fields_ = [
        ("cb", ctypes.POINTER(csinn_callback)),
        ("name", ctypes.c_char_p),
        ("layout", ctypes.c_int32),
        ("api", ctypes.c_int32),
        ("quant_type", ctypes.c_int32),
        ("sess", ctypes.POINTER(csinn_session)),
    ]


class csinn_instance_norm_params(ctypes.Structure):
    _fields_ = [
        ("base", csinn_params_base),
        ("epsilon", ctypes.c_float),
    ]


class conv_extra(ctypes.Structure):
    _fields_ = [
        ("kernel_tm", ctypes.POINTER(csinn_tensor)),
        ("conv_mode", ctypes.c_int32),
        ("fuse_zp2bias", ctypes.c_int32),
    ]


class csinn_conv2d_params(ctypes.Structure):
    _fields_ = [
        ("base", csinn_params_base),
        ("group", ctypes.c_int32),
        ("stride_height", ctypes.c_int32),
        ("stride_width", ctypes.c_int32),
        ("pad_top", ctypes.c_int32),
        ("pad_left", ctypes.c_int32),
        ("pad_down", ctypes.c_int32),
        ("pad_right", ctypes.c_int32),
        ("dilation_height", ctypes.c_int32),
        ("dilation_width", ctypes.c_int32),
        ("out_pad_height", ctypes.c_int32),
        ("out_pad_width", ctypes.c_int32),
        ("conv_extra", conv_extra),
    ]


def create_folder(folder_path):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


def compile_so(src_file, path, platform="x86_ref"):
    target_op = os.path.splitext(os.path.basename(src_file))[0]

    shl_path = "../../install_nn2"
    shl_include = f"{shl_path}/include"
    if platform == "x86_ref":
        compiler = "gcc"
        lib_path = f"{shl_path}/lib"
        lib_name = "shl_ref_x86"
    else:
        assert 0, Exception("TODO: More platforms need to be added.")

    # compile
    o_path = os.path.join(path, f"{target_op}.o")
    compile_cmd = compiler
    compile_cmd += " -c -O2 -g -fpic"
    compile_cmd += f" -I{shl_include}"
    compile_cmd += f" {src_file}"
    compile_cmd += f" -o {o_path}"
    subprocess.run(compile_cmd, shell=True, check=True)

    # link
    link_cmd = compiler
    so_path = os.path.join(path, f"{target_op}.so")
    link_cmd += " -O2 -g -fpic -shared  -lz -lstdc++ -lm"
    link_cmd += f" {o_path}"
    link_cmd += f" -L{lib_path}"
    link_cmd += f" -l{lib_name}"
    link_cmd += f" -o {so_path}"
    subprocess.run(link_cmd, shell=True, check=True)

    return so_path


class BuildSo:
    def __init__(self, src_file):
        # compile so
        self.src_file = src_file
        self.out_path = os.path.splitext(os.path.basename(src_file))[0]

    def __enter__(self):
        create_folder(self.out_path)
        self.so_path = compile_so(self.src_file, self.out_path, "x86_ref")
        return self

    def __exit__(self, *args):
        # remve tmp dir
        os.system("rm -rf " + self.out_path)
