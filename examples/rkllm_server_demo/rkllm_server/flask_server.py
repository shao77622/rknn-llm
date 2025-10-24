import ctypes
import sys
import os
import subprocess
import resource
import threading
import time
import argparse
import json
from flask import Flask, request, jsonify, Response
import re
import base64
import io
import numpy as np
import cv2

app = Flask(__name__)

# Set the dynamic library path
rkllm_lib = ctypes.CDLL('lib/librkllmrt.so')
rknn_lib = ctypes.CDLL('lib/librknnrt.so')
# Define the structures from the library
RKLLM_Handle_t = ctypes.c_void_p
userdata = ctypes.c_void_p(None)

LLMCallState = ctypes.c_int
LLMCallState.RKLLM_RUN_NORMAL  = 0
LLMCallState.RKLLM_RUN_WAITING  = 1
LLMCallState.RKLLM_RUN_FINISH  = 2
LLMCallState.RKLLM_RUN_ERROR   = 3

RKLLMInputType = ctypes.c_int
RKLLMInputType.RKLLM_INPUT_PROMPT      = 0
RKLLMInputType.RKLLM_INPUT_TOKEN       = 1
RKLLMInputType.RKLLM_INPUT_EMBED       = 2
RKLLMInputType.RKLLM_INPUT_MULTIMODAL  = 3

RKLLMInferMode = ctypes.c_int
RKLLMInferMode.RKLLM_INFER_GENERATE = 0
RKLLMInferMode.RKLLM_INFER_GET_LAST_HIDDEN_LAYER = 1
RKLLMInferMode.RKLLM_INFER_GET_LOGITS = 2
class RKLLMExtendParam(ctypes.Structure):
    _fields_ = [
        ("base_domain_id", ctypes.c_int32),
        ("embed_flash", ctypes.c_int8),
        ("enabled_cpus_num", ctypes.c_int8),
        ("enabled_cpus_mask", ctypes.c_uint32),
        ("n_batch", ctypes.c_uint8),
        ("use_cross_attn", ctypes.c_int8),
        ("reserved", ctypes.c_uint8 * 104)
    ]

class RKLLMParam(ctypes.Structure):
    _fields_ = [
        ("model_path", ctypes.c_char_p),
        ("max_context_len", ctypes.c_int32),
        ("max_new_tokens", ctypes.c_int32),
        ("top_k", ctypes.c_int32),
        ("n_keep", ctypes.c_int32),
        ("top_p", ctypes.c_float),
        ("temperature", ctypes.c_float),
        ("repeat_penalty", ctypes.c_float),
        ("frequency_penalty", ctypes.c_float),
        ("presence_penalty", ctypes.c_float),
        ("mirostat", ctypes.c_int32),
        ("mirostat_tau", ctypes.c_float),
        ("mirostat_eta", ctypes.c_float),
        ("skip_special_token", ctypes.c_bool),
        ("is_async", ctypes.c_bool),
        ("img_start", ctypes.c_char_p),
        ("img_end", ctypes.c_char_p),
        ("img_content", ctypes.c_char_p),
        ("extend_param", RKLLMExtendParam),
    ]

class RKLLMLoraAdapter(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_path", ctypes.c_char_p),
        ("lora_adapter_name", ctypes.c_char_p),
        ("scale", ctypes.c_float)
    ]

class RKLLMEmbedInput(ctypes.Structure):
    _fields_ = [
        ("embed", ctypes.POINTER(ctypes.c_float)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMTokenInput(ctypes.Structure):
    _fields_ = [
        ("input_ids", ctypes.POINTER(ctypes.c_int32)),
        ("n_tokens", ctypes.c_size_t)
    ]

class RKLLMMultiModelInput(ctypes.Structure):
    _fields_ = [
        ("prompt", ctypes.c_char_p),
        ("image_embed", ctypes.POINTER(ctypes.c_float)),
        ("n_image_tokens", ctypes.c_size_t),
        ("n_image", ctypes.c_size_t),
        ("image_width", ctypes.c_size_t),
        ("image_height", ctypes.c_size_t)
    ]

class RKLLMInputUnion(ctypes.Union):
    _fields_ = [
        ("prompt_input", ctypes.c_char_p),
        ("embed_input", RKLLMEmbedInput),
        ("token_input", RKLLMTokenInput),
        ("multimodal_input", RKLLMMultiModelInput)
    ]

class RKLLMInput(ctypes.Structure):
    _fields_ = [
        ("role", ctypes.c_char_p),
        ("enable_thinking", ctypes.c_bool),
        ("input_type", RKLLMInputType),
        ("input_data", RKLLMInputUnion)
    ]

class RKLLMLoraParam(ctypes.Structure):
    _fields_ = [
        ("lora_adapter_name", ctypes.c_char_p)
    ]

class RKLLMPromptCacheParam(ctypes.Structure):
    _fields_ = [
        ("save_prompt_cache", ctypes.c_int),
        ("prompt_cache_path", ctypes.c_char_p)
    ]

class RKLLMInferParam(ctypes.Structure):
    _fields_ = [
        ("mode", RKLLMInferMode),
        ("lora_params", ctypes.POINTER(RKLLMLoraParam)),
        ("prompt_cache_params", ctypes.POINTER(RKLLMPromptCacheParam)),
        ("keep_history", ctypes.c_int)
    ]

class RKLLMResultLastHiddenLayer(ctypes.Structure):
    _fields_ = [
        ("hidden_states", ctypes.POINTER(ctypes.c_float)),
        ("embd_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMResultLogits(ctypes.Structure):
    _fields_ = [
        ("logits", ctypes.POINTER(ctypes.c_float)),
        ("vocab_size", ctypes.c_int),
        ("num_tokens", ctypes.c_int)
    ]

class RKLLMPerfStat(ctypes.Structure):
    _fields_ = [
        ("prefill_time_ms", ctypes.c_float),
        ("prefill_tokens", ctypes.c_int),
        ("generate_time_ms", ctypes.c_float),
        ("generate_tokens", ctypes.c_int),
        ("memory_usage_mb", ctypes.c_float)
    ]

class RKLLMResult(ctypes.Structure):
    _fields_ = [
        ("text", ctypes.c_char_p),
        ("token_id", ctypes.c_int),
        ("last_hidden_layer", RKLLMResultLastHiddenLayer),
        ("logits", RKLLMResultLogits),
        ("perf", RKLLMPerfStat)
    ]

# RKNN related structures for image encoder
rknn_context = ctypes.c_uint64

class rknn_tensor_attr(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("n_dims", ctypes.c_uint32),
        ("dims", ctypes.c_uint32 * 16),
        ("name", ctypes.c_char * 256),
        ("n_elems", ctypes.c_uint32),
        ("size", ctypes.c_uint32),
        ("fmt", ctypes.c_int32),
        ("type", ctypes.c_int32),
        ("qnt_type", ctypes.c_int32),
        ("fl", ctypes.c_int8),
        ("zp", ctypes.c_int32),
        ("scale", ctypes.c_float),
        ("w_stride", ctypes.c_uint32),
        ("size_with_stride", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("h_stride", ctypes.c_uint32),
    ]

class rknn_input(ctypes.Structure):
    _fields_ = [
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
        ("pass_through", ctypes.c_uint8),
        ("type", ctypes.c_int32),
        ("fmt", ctypes.c_int32),
    ]

class rknn_output(ctypes.Structure):
    _fields_ = [
        ("want_float", ctypes.c_uint8),
        ("is_prealloc", ctypes.c_uint8),
        ("index", ctypes.c_uint32),
        ("buf", ctypes.c_void_p),
        ("size", ctypes.c_uint32),
    ]

class rknn_input_output_num(ctypes.Structure):
    _fields_ = [
        ("n_input", ctypes.c_uint32),
        ("n_output", ctypes.c_uint32),
    ]

class RKNNAppContext:
    def __init__(self):
        self.rknn_ctx = rknn_context(0)
        self.io_num = rknn_input_output_num()
        self.input_attrs = None
        self.output_attrs = None
        self.model_channel = 0
        self.model_width = 0
        self.model_height = 0
        self.model_image_token = 0
        self.model_embed_size = 0

# Create a lock to control multi-user access to the server.
lock = threading.Lock()

# Create a global variable to indicate whether the server is currently in a blocked state.
is_blocking = False

# Define global variables to store the callback function output for displaying in the Gradio interface
system_prompt = ''
global_text = []
global_state = -1
split_byte_data = bytes(b"") # Used to store the segmented byte data

recevied_messages = []

# Define the callback function
def callback_impl(result, userdata, state):
    global global_text, global_state, split_byte_data
    if state == LLMCallState.RKLLM_RUN_FINISH:
        global_state = state
        print("\n")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_ERROR:
        global_state = state
        print("run error")
        sys.stdout.flush()
    elif state == LLMCallState.RKLLM_RUN_NORMAL:
        global_state = state
        global_text += result.contents.text.decode('utf-8')
    return 0
    

# Connect the callback function between the Python side and the C++ side
callback_type = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(RKLLMResult), ctypes.c_void_p, ctypes.c_int)
callback = callback_type(callback_impl)

# Image processing functions
def preprocess_image(image_data, target_width, target_height):
    """Preprocess image: decode -> resize"""
    if isinstance(image_data, str):
        if image_data.startswith('data:image'):
            image_data = image_data.split(',', 1)[1]
        image_bytes = base64.b64decode(image_data)
    else:
        image_bytes = image_data
    
    img_array = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Failed to decode image")
    
    img_resized = cv2.resize(img, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
    return img_resized

# RKNN Image Encoder functions
def init_imgenc(model_path, core_num=3):
    """Initialize RKNN image encoder model"""
    app_ctx = RKNNAppContext()
    
    # RKNN API constants
    RKNN_SUCC = 0
    RKNN_QUERY_IN_OUT_NUM = 0
    RKNN_QUERY_INPUT_ATTR = 1
    RKNN_QUERY_OUTPUT_ATTR = 2
    RKNN_NPU_CORE_AUTO = 0
    RKNN_NPU_CORE_0_1 = 3
    RKNN_NPU_CORE_0_1_2 = 7
    RKNN_TENSOR_NCHW = 0
    
    # Read model file into memory
    try:
        with open(model_path, 'rb') as f:
            model_data = f.read()
        model_size = len(model_data)
    except Exception as e:
        print(f"Failed to read model file: {e}")
        return None
    
    model_buffer = (ctypes.c_ubyte * model_size).from_buffer_copy(model_data)
    
    rknn_init = rknn_lib.rknn_init
    rknn_init.argtypes = [ctypes.POINTER(rknn_context), ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
    rknn_init.restype = ctypes.c_int
    
    ret = rknn_init(ctypes.byref(app_ctx.rknn_ctx), ctypes.cast(model_buffer, ctypes.c_void_p), model_size, 0, None)
    if ret != RKNN_SUCC:
        print(f"rknn_init failed! ret={ret}")
        return None
    
    # Set core mask
    rknn_set_core_mask = rknn_lib.rknn_set_core_mask
    rknn_set_core_mask.argtypes = [rknn_context, ctypes.c_int32]
    rknn_set_core_mask.restype = ctypes.c_int
    
    if core_num == 2:
        rknn_set_core_mask(app_ctx.rknn_ctx, RKNN_NPU_CORE_0_1)
    elif core_num == 3:
        rknn_set_core_mask(app_ctx.rknn_ctx, RKNN_NPU_CORE_0_1_2)
    else:
        rknn_set_core_mask(app_ctx.rknn_ctx, RKNN_NPU_CORE_AUTO)
    
    # Query input/output number
    rknn_query = rknn_lib.rknn_query
    rknn_query.argtypes = [rknn_context, ctypes.c_int32, ctypes.c_void_p, ctypes.c_uint32]
    rknn_query.restype = ctypes.c_int
    
    ret = rknn_query(app_ctx.rknn_ctx, RKNN_QUERY_IN_OUT_NUM, ctypes.byref(app_ctx.io_num), ctypes.sizeof(app_ctx.io_num))
    if ret != RKNN_SUCC:
        print(f"rknn_query IN_OUT_NUM failed! ret={ret}")
        return None
    
    print(f"RKNN model input num: {app_ctx.io_num.n_input}, output num: {app_ctx.io_num.n_output}")
    
    # Query input attributes
    app_ctx.input_attrs = (rknn_tensor_attr * app_ctx.io_num.n_input)()
    for i in range(app_ctx.io_num.n_input):
        app_ctx.input_attrs[i].index = i
        ret = rknn_query(app_ctx.rknn_ctx, RKNN_QUERY_INPUT_ATTR, ctypes.byref(app_ctx.input_attrs[i]), ctypes.sizeof(rknn_tensor_attr))
        if ret != RKNN_SUCC:
            print(f"rknn_query INPUT_ATTR {i} failed! ret={ret}")
            return None
    
    # Query output attributes
    app_ctx.output_attrs = (rknn_tensor_attr * app_ctx.io_num.n_output)()
    for i in range(app_ctx.io_num.n_output):
        app_ctx.output_attrs[i].index = i
        ret = rknn_query(app_ctx.rknn_ctx, RKNN_QUERY_OUTPUT_ATTR, ctypes.byref(app_ctx.output_attrs[i]), ctypes.sizeof(rknn_tensor_attr))
        if ret != RKNN_SUCC:
            print(f"rknn_query OUTPUT_ATTR {i} failed! ret={ret}")
            return None
    
    # Set context parameters from output tensor
    if app_ctx.output_attrs[0].n_dims == 3:
        app_ctx.model_image_token = app_ctx.output_attrs[0].dims[1]
        app_ctx.model_embed_size = app_ctx.output_attrs[0].dims[2]
    else:
        # app_ctx.model_image_token = app_ctx.output_attrs[0].dims[0]
        app_ctx.model_image_token = app_ctx.output_attrs[0].dims[0]
        app_ctx.model_embed_size = app_ctx.output_attrs[0].dims[1]
    
    if app_ctx.input_attrs[0].fmt == RKNN_TENSOR_NCHW:
        app_ctx.model_channel = app_ctx.input_attrs[0].dims[1]
        app_ctx.model_height = app_ctx.input_attrs[0].dims[2]
        app_ctx.model_width = app_ctx.input_attrs[0].dims[3]
    else:
        app_ctx.model_height = app_ctx.input_attrs[0].dims[1]
        app_ctx.model_width = app_ctx.input_attrs[0].dims[2]
        app_ctx.model_channel = app_ctx.input_attrs[0].dims[3]
    
    print(f"Vision encoder initialized: {app_ctx.model_width}x{app_ctx.model_height}, "
          f"tokens={app_ctx.model_image_token}, embed_size={app_ctx.model_embed_size}")
    
    return app_ctx

def run_imgenc(app_ctx, img_data):
    """Run RKNN image encoder inference"""
    RKNN_SUCC = 0
    RKNN_TENSOR_FLOAT32 = 0
    RKNN_TENSOR_NHWC = 1
    
    # Convert to float32 and add batch dimension (1, H, W, C)
    img_float = img_data.astype(np.float32)
    img_batch = img_float[np.newaxis, :, :, :]
    
    # Prepare input
    inputs = (rknn_input * 1)()
    inputs[0].index = 0
    inputs[0].type = RKNN_TENSOR_FLOAT32
    inputs[0].fmt = RKNN_TENSOR_NHWC
    inputs[0].size = img_batch.nbytes
    inputs[0].buf = img_batch.ctypes.data_as(ctypes.c_void_p)
    inputs[0].pass_through = 0
    
    # Set inputs
    rknn_inputs_set = rknn_lib.rknn_inputs_set
    rknn_inputs_set.argtypes = [rknn_context, ctypes.c_uint32, ctypes.POINTER(rknn_input)]
    rknn_inputs_set.restype = ctypes.c_int
    
    ret = rknn_inputs_set(app_ctx.rknn_ctx, 1, inputs)
    if ret != RKNN_SUCC:
        print(f"rknn_inputs_set failed! ret={ret}")
        return None
    
    # Run inference
    rknn_run = rknn_lib.rknn_run
    rknn_run.argtypes = [rknn_context, ctypes.c_void_p]
    rknn_run.restype = ctypes.c_int
    
    ret = rknn_run(app_ctx.rknn_ctx, None)
    if ret != RKNN_SUCC:
        print(f"rknn_run failed! ret={ret}")
        return None
    
    # Get outputs
    outputs = (rknn_output * 1)()
    outputs[0].want_float = 1
    outputs[0].is_prealloc = 0
    
    rknn_outputs_get = rknn_lib.rknn_outputs_get
    rknn_outputs_get.argtypes = [rknn_context, ctypes.c_uint32, ctypes.POINTER(rknn_output), ctypes.c_void_p]
    rknn_outputs_get.restype = ctypes.c_int
    
    ret = rknn_outputs_get(app_ctx.rknn_ctx, 1, outputs, None)
    if ret != RKNN_SUCC:
        print(f"rknn_outputs_get failed! ret={ret}")
        return None
    
    # Copy output data
    output_size = app_ctx.model_image_token * app_ctx.model_embed_size
    output_array = np.ctypeslib.as_array(
        ctypes.cast(outputs[0].buf, ctypes.POINTER(ctypes.c_float)),
        shape=(output_size,)
    ).copy()
    
    # Release outputs
    rknn_outputs_release = rknn_lib.rknn_outputs_release
    rknn_outputs_release.argtypes = [rknn_context, ctypes.c_uint32, ctypes.POINTER(rknn_output)]
    rknn_outputs_release.restype = ctypes.c_int
    rknn_outputs_release(app_ctx.rknn_ctx, 1, outputs)
    
    return output_array

def release_imgenc(app_ctx):
    """Release RKNN image encoder resources"""
    if app_ctx and app_ctx.rknn_ctx:
        rknn_destroy = rknn_lib.rknn_destroy
        rknn_destroy.argtypes = [rknn_context]
        rknn_destroy.restype = ctypes.c_int
        rknn_destroy(app_ctx.rknn_ctx)
        app_ctx.rknn_ctx = rknn_context(0)
    return 0

# Define the RKLLM class, which includes initialization, inference, and release operations for the RKLLM model in the dynamic library
class RKLLM(object):
    def __init__(self, model_path, lora_model_path = None, prompt_cache_path = None, platform = "rk3588", max_context_len = 4096, max_new_tokens = 4096, use_cross_attn = 0, base_domain_id = 0):
        rkllm_param = RKLLMParam()
        rkllm_param.model_path = bytes(model_path, 'utf-8')

        # max_context_len: 最大上下文长度（即输入+输出总token数的最大值，影响模型能记住的历史内容长度）
        rkllm_param.max_context_len = max_context_len
        # max_new_tokens: 推理生成的最大新token数量（即每次生成回复的最大长度）
        rkllm_param.max_new_tokens = max_new_tokens
        # skip_special_token: 是否跳过特殊token，True表示输出时去除特殊字符
        rkllm_param.skip_special_token = True
        # n_keep: 生成时保留的前缀token数，一般设为-1表示全部可变
        rkllm_param.n_keep = -1
        # top_k: 采样时从概率最高的top_k个token中抽样，大值更随机，小值更确定
        rkllm_param.top_k = 1
        # top_p: nucleus采样概率阈值，通常0.7~0.95，越低越确定
        rkllm_param.top_p = 0.7
        # temperature: 采样温度，越高越随机，越低越保守（0.1-1.0），0.3偏保守
        rkllm_param.temperature = 0.5
        # repeat_penalty: 重复惩罚系数，防止生成重复内容，通常1.1~1.3
        rkllm_param.repeat_penalty = 1.1
        # frequency_penalty: 按已有频率惩罚已出现词语，抑制常用词
        rkllm_param.frequency_penalty = 0.0
        # presence_penalty: 按是否出现过惩罚，增强多样性
        rkllm_param.presence_penalty = 0.0
        # mirostat系列参数：用于自适应生成控制输出困惑度（见mirostat算法论文）
        # mirostat: 0表示不用，1/2表示不同版本自适应采样
        rkllm_param.mirostat = 0
        # 期望困惑度
        rkllm_param.mirostat_tau = 5.0  
        # 调整速率
        rkllm_param.mirostat_eta = 0.1  

        # is_async: 是否异步推理（通常设为False，同步推理即可）
        rkllm_param.is_async = False

        # Set multimodal parameters (required for vision models like Qwen2-VL)
        # These tokens must match the model's training configuration
        rkllm_param.img_start = "<|vision_start|>".encode('utf-8')
        rkllm_param.img_end = "<|vision_end|>".encode('utf-8')
        rkllm_param.img_content = "<|image_pad|>".encode('utf-8')

        rkllm_param.extend_param.base_domain_id = base_domain_id
        rkllm_param.extend_param.embed_flash = 1
        rkllm_param.extend_param.n_batch = 1
        rkllm_param.extend_param.use_cross_attn = use_cross_attn
        rkllm_param.extend_param.enabled_cpus_num = 4
        if platform.lower() in ["rk3576", "rk3588"]:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 4)|(1 << 5)|(1 << 6)|(1 << 7)
        else:
            rkllm_param.extend_param.enabled_cpus_mask = (1 << 0)|(1 << 1)|(1 << 2)|(1 << 3)

        self.handle = RKLLM_Handle_t()

        self.rkllm_init = rkllm_lib.rkllm_init
        self.rkllm_init.argtypes = [ctypes.POINTER(RKLLM_Handle_t), ctypes.POINTER(RKLLMParam), callback_type]
        self.rkllm_init.restype = ctypes.c_int
        ret = self.rkllm_init(ctypes.byref(self.handle), ctypes.byref(rkllm_param), callback)
        if (ret != 0):
            print("\nrkllm init failed\n")
            exit(0)
        else:
            print("\nrkllm init success!\n")

        self.rkllm_run = rkllm_lib.rkllm_run
        self.rkllm_run.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMInput), ctypes.POINTER(RKLLMInferParam), ctypes.c_void_p]
        self.rkllm_run.restype = ctypes.c_int
        
        self.set_chat_template = rkllm_lib.rkllm_set_chat_template
        self.set_chat_template.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.set_chat_template.restype = ctypes.c_int
        
        self.set_function_tools_ = rkllm_lib.rkllm_set_function_tools
        self.set_function_tools_.argtypes = [RKLLM_Handle_t, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p]
        self.set_function_tools_.restype = ctypes.c_int
        
        # system_prompt = "<|im_start|>system You are a helpful assistant. <|im_end|>"
        # prompt_prefix = "<|im_start|>user"
        # prompt_postfix = "<|im_end|><|im_start|>assistant"
        # self.set_chat_template(self.handle, ctypes.c_char_p(system_prompt.encode('utf-8')), ctypes.c_char_p(prompt_prefix.encode('utf-8')), ctypes.c_char_p(prompt_postfix.encode('utf-8')))

        self.rkllm_destroy = rkllm_lib.rkllm_destroy
        self.rkllm_destroy.argtypes = [RKLLM_Handle_t]
        self.rkllm_destroy.restype = ctypes.c_int
        
        self.rkllm_abort = rkllm_lib.rkllm_abort

        rkllm_lora_params = None
        if lora_model_path:
            lora_adapter_name = "test"
            lora_adapter = RKLLMLoraAdapter()
            ctypes.memset(ctypes.byref(lora_adapter), 0, ctypes.sizeof(RKLLMLoraAdapter))
            lora_adapter.lora_adapter_path = ctypes.c_char_p((lora_model_path).encode('utf-8'))
            lora_adapter.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode('utf-8'))
            lora_adapter.scale = 1.0

            rkllm_load_lora = rkllm_lib.rkllm_load_lora
            rkllm_load_lora.argtypes = [RKLLM_Handle_t, ctypes.POINTER(RKLLMLoraAdapter)]
            rkllm_load_lora.restype = ctypes.c_int
            rkllm_load_lora(self.handle, ctypes.byref(lora_adapter))
            rkllm_lora_params = RKLLMLoraParam()
            rkllm_lora_params.lora_adapter_name = ctypes.c_char_p((lora_adapter_name).encode('utf-8'))
        
        self.rkllm_infer_params = RKLLMInferParam()
        ctypes.memset(ctypes.byref(self.rkllm_infer_params), 0, ctypes.sizeof(RKLLMInferParam))
        self.rkllm_infer_params.mode = RKLLMInferMode.RKLLM_INFER_GENERATE
        self.rkllm_infer_params.lora_params = ctypes.pointer(rkllm_lora_params) if rkllm_lora_params else None
        self.rkllm_infer_params.keep_history = 0

        self.prompt_cache_path = None
        if prompt_cache_path:
            self.prompt_cache_path = prompt_cache_path

            rkllm_load_prompt_cache = rkllm_lib.rkllm_load_prompt_cache
            rkllm_load_prompt_cache.argtypes = [RKLLM_Handle_t, ctypes.c_char_p]
            rkllm_load_prompt_cache.restype = ctypes.c_int
            rkllm_load_prompt_cache(self.handle, ctypes.c_char_p((prompt_cache_path).encode('utf-8')))
        
        self.tools = None
            
    def set_function_tools(self, system_prompt, tools, tool_response_str):
        if self.tools is None or not self.tools == tools:
            self.tools = tools
            self.set_function_tools_(self.handle, ctypes.c_char_p(system_prompt.encode('utf-8')), ctypes.c_char_p(tools.encode('utf-8')),  ctypes.c_char_p(tool_response_str.encode('utf-8')))

    def run(self, *param):
        if len(param) == 3:
            # Text-only mode: role, enable_thinking, prompt
            role, enable_thinking, prompt = param
            rkllm_input = RKLLMInput()
            rkllm_input.role = role.encode('utf-8') if role is not None else "user".encode('utf-8')
            rkllm_input.enable_thinking = ctypes.c_bool(enable_thinking if enable_thinking is not None else False)
            rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_PROMPT
            rkllm_input.input_data.prompt_input = ctypes.c_char_p(prompt.encode('utf-8'))
            self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)
        elif len(param) == 8:
            # Multimodal mode: role, enable_thinking, prompt, image_embed, n_image_tokens, n_image, image_width, image_height
            role, enable_thinking, prompt, image_embed, n_image_tokens, n_image, image_width, image_height = param
            rkllm_input = RKLLMInput()
            rkllm_input.role = role.encode('utf-8') if role is not None else "user".encode('utf-8')
            rkllm_input.enable_thinking = ctypes.c_bool(enable_thinking if enable_thinking is not None else False)
            rkllm_input.input_type = RKLLMInputType.RKLLM_INPUT_MULTIMODAL
            rkllm_input.input_data.multimodal_input.prompt = ctypes.c_char_p(prompt.encode('utf-8'))
            rkllm_input.input_data.multimodal_input.image_embed = image_embed.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            rkllm_input.input_data.multimodal_input.n_image_tokens = n_image_tokens
            rkllm_input.input_data.multimodal_input.n_image = n_image
            rkllm_input.input_data.multimodal_input.image_width = image_width
            rkllm_input.input_data.multimodal_input.image_height = image_height
            self.rkllm_run(self.handle, ctypes.byref(rkllm_input), ctypes.byref(self.rkllm_infer_params), None)
        else:
            raise ValueError(f"Invalid number of parameters: {len(param)}")
        return
    
    def abort(self):
        return self.rkllm_abort(self.handle)
    
    def release(self):
        self.rkllm_destroy(self.handle)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rkllm_model_path', type=str, required=True, help='Absolute path of the converted RKLLM model on the Linux board;')
    parser.add_argument('--target_platform', type=str, required=True, help='Target platform: e.g., rk3588/rk3576;')
    parser.add_argument('--lora_model_path', type=str, help='Absolute path of the lora_model on the Linux board;')
    parser.add_argument('--prompt_cache_path', type=str, help='Absolute path of the prompt_cache file on the Linux board;')
    parser.add_argument('--vision_model_path', type=str, help='Absolute path of the RKNN vision encoder model for multimodal inference;')
    parser.add_argument('--npu_core_num', type=int, default=3, help='Number of NPU cores to use for vision encoder (1/2/3);')
    parser.add_argument('--max_context_len', type=int, default=4096, help='Maximum context length (reduce if memory is limited);')
    parser.add_argument('--max_new_tokens', type=int, default=4096, help='Maximum new tokens (reduce if memory is limited);')
    parser.add_argument('--use_cross_attn', type=int, default=0, help='Use cross attention for multimodal (0 or 1);')
    parser.add_argument('--base_domain_id', type=int, default=None, help='Base domain ID (auto-detect: 1 for multimodal, 0 for text-only);')
    args = parser.parse_args()

    if not os.path.exists(args.rkllm_model_path):
        print("Error: Please provide the correct rkllm model path, and ensure it is the absolute path on the board.")
        sys.stdout.flush()
        exit()

    if not (args.target_platform in ["rk3588", "rk3576", "rv1126b", "rk3562"]):
        print("Error: Please specify the correct target platform: rk3588/rk3576/rv1126b/rk3562.")
        sys.stdout.flush()
        exit()

    if args.lora_model_path:
        if not os.path.exists(args.lora_model_path):
            print("Error: Please provide the correct lora_model path, and advise it is the absolute path on the board.")
            sys.stdout.flush()
            exit()

    if args.prompt_cache_path:
        if not os.path.exists(args.prompt_cache_path):
            print("Error: Please provide the correct prompt_cache_file path, and advise it is the absolute path on the board.")
            sys.stdout.flush()
            exit()

    # Fix frequency
    command = "sudo bash fix_freq_{}.sh".format(args.target_platform)
    subprocess.run(command, shell=True)

    # Set resource limit
    resource.setrlimit(resource.RLIMIT_NOFILE, (102400, 102400))

    # Initialize vision encoder FIRST (if provided) to ensure NPU memory is available
    vision_encoder = None
    if args.vision_model_path:
        if not os.path.exists(args.vision_model_path):
            print("Error: Vision model path not found:", args.vision_model_path)
            exit()
        print("Initializing vision encoder...")
        vision_encoder = init_imgenc(args.vision_model_path, args.npu_core_num)
        if vision_encoder is None:
            print("Error: Failed to initialize vision encoder.")
            exit()
    else:
        print("No vision model provided, running in text-only mode")
    
    # Initialize RKLLM model AFTER vision encoder
    print("Initializing RKLLM...")
    base_domain_id = 1 if vision_encoder else 0 if args.base_domain_id is None else args.base_domain_id
    
    model_path = args.rkllm_model_path
    rkllm_model = RKLLM(
        model_path, 
        args.lora_model_path, 
        args.prompt_cache_path, 
        args.target_platform,
        max_context_len=args.max_context_len,
        max_new_tokens=args.max_new_tokens,
        use_cross_attn=args.use_cross_attn,
        base_domain_id=base_domain_id
    )

    # Helper function to process messages with images
    def process_multimodal_messages(messages, vision_encoder):
        """Extract and process images from messages"""
        processed_messages = []
        image_embeds_list = []
        
        for message in messages:
            if not isinstance(message.get('content'), list):
                # Text-only message
                processed_messages.append(message)
                continue
            
            # Process multimodal content
            text_parts = []
            image_parts = []
            
            for content_item in message['content']:
                if content_item.get('type') == 'text':
                    text_parts.append(content_item.get('text', ''))
                elif content_item.get('type') == 'image_url':
                    image_url = content_item.get('image_url', {})
                    image_data = image_url.get('url', '')
                    if image_data:
                        image_parts.append(image_data)
                elif content_item.get('type') == 'image':
                    # Direct base64 image
                    image_data = content_item.get('image', '')
                    if image_data:
                        image_parts.append(image_data)
            
            # Process images if vision encoder is available
            if image_parts and vision_encoder:
                for img_data in image_parts:
                    try:
                        # Preprocess image
                        img_array = preprocess_image(img_data, vision_encoder.model_width, vision_encoder.model_height)
                        # Run vision encoder
                        img_embed = run_imgenc(vision_encoder, img_array)
                        if img_embed is not None:
                            image_embeds_list.append(img_embed)
                            # Add image placeholder to text
                            text_parts.insert(0, '<image>')
                    except Exception as e:
                        print(f"Error processing image: {e}")
                        sys.stdout.flush()
            
            # Create processed message with combined text
            processed_message = message.copy()
            processed_message['content'] = ' '.join(text_parts)
            processed_messages.append(processed_message)
        
        return processed_messages, image_embeds_list

    # Create a function to receive data sent by the user using a request
    @app.route('/rkllm_chat', methods=['POST'])
    def receive_message():
        # Link global variables to retrieve the output information from the callback function
        global global_text, global_state, system_prompt, recevied_messages
        global is_blocking

        # If the server is in a blocking state, return a specific response.
        if is_blocking or global_state==0:
            return jsonify({'status': 'error', 'message': 'RKLLM_Server is busy! Maybe you can try again later.'}), 503
        
        lock.acquire()
        try:
            # Set the server to a blocking state.
            is_blocking = True

            # Get JSON data from the POST request.
            data = request.json
            if data and 'messages' in data:
                # Reset global variables.
                global_text = []
                global_state = -1

                # Define the structure for the returned response.
                rkllm_responses = {
                    "id": "rkllm_chat",
                    "object": "rkllm_chat",
                    "created": None,
                    "choices": [],
                    "usage": {
                    "prompt_tokens": None,
                    "completion_tokens": None,
                    "total_tokens": None
                    }
                }

                if not "stream" in data.keys() or data["stream"] == False:
                    # Process the received data here.
                    messages = data['messages']
                    enable_thinking = data.get('enable_thinking')
                    TOOLS = data.get('tools')
                    print("Received messages:", messages)
                    
                    # Process multimodal messages if vision encoder is available
                    image_embeds_list = []
                    if vision_encoder:
                        messages, image_embeds_list = process_multimodal_messages(messages, vision_encoder)
                    
                    input_prompt = []
                    for index, message in enumerate(messages):

                        if message not in recevied_messages and TOOLS is not None:
                            recevied_messages.append(message)
                        else:
                            if index >= 1:
                                print('skip recevied_messages')
                                continue
                        
                        if message['role'] == 'system':
                            system_prompt = message['content']
                            print('skip system messages')
                            continue

                        if message['role'] == 'assistant':
                            print('skip assistant messages')
                            continue
                        
                        if message['role'] == 'tool':
                            if not isinstance(input_prompt, list):
                                input_prompt = []
                            input_prompt.append(message['content'])
                            if index < len(messages) - 1:
                                continue
                            
                        if message['role'] == 'user':
                            input_prompt = message['content']
                        elif message['role'] == 'tool':
                            input_prompt = json.dumps(input_prompt)
                            recevied_messages.clear()
                        else:
                            print("role setting error")
                            
                        rkllm_output = ""
                        
                        # Create a thread for model inference.
                        if TOOLS is not None:
                            rkllm_model.set_function_tools(system_prompt=system_prompt, tools=json.dumps(TOOLS),  tool_response_str="tool_response")
                        else:
                            # Clear function tools when tools is None
                            if rkllm_model.tools is not None:
                                rkllm_model.tools = None
                                # Reset with empty tools to clear previous configuration
                                rkllm_model.set_function_tools_(rkllm_model.handle, ctypes.c_char_p("".encode('utf-8')), ctypes.c_char_p("".encode('utf-8')), ctypes.c_char_p("".encode('utf-8')))
                        
                        # Check if we have image embeds (multimodal mode)
                        if image_embeds_list:
                            # Concatenate all image embeddings
                            combined_embeds = np.concatenate(image_embeds_list, axis=0)
                            n_images = len(image_embeds_list)
                            n_image_tokens = vision_encoder.model_image_token
                            image_width = vision_encoder.model_width
                            image_height = vision_encoder.model_height
                            
                            model_thread = threading.Thread(
                                target=rkllm_model.run,
                                args=(message['role'], enable_thinking, input_prompt, 
                                      combined_embeds, n_image_tokens, n_images, image_width, image_height)
                            )
                        else:
                            # Text-only mode
                            model_thread = threading.Thread(target=rkllm_model.run, args=(message['role'], enable_thinking, input_prompt, ))
                        model_thread.start()

                        # Wait for the model to finish running and periodically check the inference thread of the model.
                        model_thread_finished = False
                        while not model_thread_finished:
                            while len(global_text) > 0:
                                rkllm_output += global_text.pop(0)
                                time.sleep(0.005)

                            model_thread.join(timeout=0.005)
                            model_thread_finished = not model_thread.is_alive()
                    
                        rkllm_responses["choices"].append(
                            {"index": index,
                            "message": {
                                "role": "assistant",
                                "content": rkllm_output,
                            },
                            "logprobs": None,
                            "finish_reason": "stop"
                            }
                        )
                    return jsonify(rkllm_responses), 200
                else:
                    messages = data['messages']
                    enable_thinking = data.get('enable_thinking')
                    TOOLS = data.get('tools')
                    print("Received messages:", messages)
                    
                    # Process multimodal messages if vision encoder is available
                    image_embeds_list = []
                    if vision_encoder:
                        messages, image_embeds_list = process_multimodal_messages(messages, vision_encoder)
                    
                    input_prompt = []
                    for index, message in enumerate(messages):
                        # print(recevied_messages)
                        if message not in recevied_messages and TOOLS is not None:
                            recevied_messages.append(message)
                        else:
                            if index >= 1:
                                print('skip recevied_messages')
                                continue
                        
                        if message['role'] == 'system':
                            system_prompt = message['content']
                            print('skip system messages')
                            continue

                        if message['role'] == 'assistant':
                            print('skip assistant messages')
                            continue
                        
                        if message['role'] == 'tool':
                            if not isinstance(input_prompt, list):
                                input_prompt = []
                            input_prompt.append(message['content'])
                            if index < len(messages) - 1:
                                continue
                            
                        if message['role'] == 'user':
                            input_prompt = message['content']
                        elif message['role'] == 'tool':
                            input_prompt = json.dumps(input_prompt)
                            recevied_messages.clear()
                        else:
                            print("role setting error")
                            
                        role = message.get('role')
                        rkllm_output = ""
                        
                        if TOOLS is not None:
                            rkllm_model.set_function_tools(system_prompt=system_prompt, tools=json.dumps(TOOLS),  tool_response_str="tool_response")
                        else:
                            # Clear function tools when tools is None
                            if rkllm_model.tools is not None:
                                rkllm_model.tools = None
                                # Reset with empty tools to clear previous configuration
                                rkllm_model.set_function_tools_(rkllm_model.handle, ctypes.c_char_p("".encode('utf-8')), ctypes.c_char_p("".encode('utf-8')), ctypes.c_char_p("".encode('utf-8')))
                        
                        def generate():
                            # Check if we have image embeds (multimodal mode)
                            if image_embeds_list:
                                # Concatenate all image embeddings
                                combined_embeds = np.concatenate(image_embeds_list, axis=0)
                                n_images = len(image_embeds_list)
                                n_image_tokens = vision_encoder.model_image_token
                                image_width = vision_encoder.model_width
                                image_height = vision_encoder.model_height
                                
                                model_thread = threading.Thread(
                                    target=rkllm_model.run,
                                    args=(role, enable_thinking, input_prompt,
                                          combined_embeds, n_image_tokens, n_images, image_width, image_height)
                                )
                            else:
                                # Text-only mode
                                model_thread = threading.Thread(target=rkllm_model.run, args=(role, enable_thinking, input_prompt, ))
                            model_thread.start()

                            model_thread_finished = False
                            while not model_thread_finished:
                                while len(global_text) > 0:
                                    rkllm_output = global_text.pop(0)

                                    rkllm_responses["choices"].append(
                                        {"index": index,
                                        "delta": {
                                            "role": "assistant",
                                            "content": rkllm_output[-1],
                                        },
                                        "logprobs": None,
                                        "finish_reason": "stop" if global_state == 1 else None,
                                        }
                                    )
                                    yield f"{json.dumps(rkllm_responses)}\n\n"

                                model_thread.join(timeout=0.005)
                                model_thread_finished = not model_thread.is_alive()

                    return Response(generate(), content_type='text/plain')
            else:
                return jsonify({'status': 'error', 'message': 'Invalid JSON data!'}), 400
        finally:
            lock.release()
            is_blocking = False
        
    # Start the Flask application.
    # app.run(host='0.0.0.0', port=8080)
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=False)

    print("====================")
    print("RKLLM model inference completed, releasing RKLLM model resources...")
    rkllm_model.release()
    if vision_encoder:
        print("Releasing vision encoder resources...")
        release_imgenc(vision_encoder)
    print("====================")
