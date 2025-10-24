import sys
import requests
import json
import re
import base64
import os
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from pathlib import Path

# Set the address of the Server.
server_url = 'http://192.168.2.30:8080/rkllm_chat'
# server_url = 'http://[240e:390:34a:ce10:24d4:f0ff:fe25:ae43]:8080/rkllm_chat'

# Create a session object.
session = requests.Session()
session.keep_alive = False  # Close the connection pool to maintain a long connection.
adapter = requests.adapters.HTTPAdapter(max_retries=5)
session.mount('https://', adapter)
session.mount('http://', adapter)

def main_demo2(is_streaming=True):
    
    ## Define the function you need to call and its description
    TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "get_current_temperature",
                "description": "Get current temperature at a location.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": 'The location to get the temperature for, in the format "City, State, Country".',
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": 'The unit to return the temperature in. Defaults to "celsius".',
                        },
                    },
                    "required": ["location"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_temperature_date",
                "description": "Get temperature at a location and date.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": 'The location to get the temperature for, in the format "City, State, Country".',
                        },
                        "date": {
                            "type": "string",
                            "description": 'The date to get the temperature for, in the format "Year-Month-Day".',
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": 'The unit to return the temperature in. Defaults to "celsius".',
                        },
                    },
                    "required": ["location", "date"],
                },
            },
        },
    ]
    
    def get_current_temperature(location: str, unit: str = "celsius"):
        """Get current temperature at a location.

        Args:
            location: The location to get the temperature for, in the format "City, State, Country".
            unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

        Returns:
            the temperature, the location, and the unit in a dict
        """
        return {
            "temperature": 26.1,
            "location": location,
            "unit": unit,
        }


    def get_temperature_date(location: str, date: str, unit: str = "celsius"):
        """Get temperature at a location and date.

        Args:
            location: The location to get the temperature for, in the format "City, State, Country".
            date: The date to get the temperature for, in the format "Year-Month-Day".
            unit: The unit to return the temperature in. Defaults to "celsius". (choices: ["celsius", "fahrenheit"])

        Returns:
            the temperature, the location, the date and the unit in a dict
        """
        return {
            "temperature": 25.9,
            "location": location,
            "date": date,
            "unit": unit,
        }

    def get_function_by_name(name):
        if name == "get_current_temperature":
            return get_current_temperature
        if name == "get_temperature_date":
            return get_temperature_date
    
    
    print("============================")
    print("This is a demo about RKLLM function-call...")
    print("============================")
    
    messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant.\n\nCurrent Date: 2024-09-30"},
    {"role": "user",  "content": "What's the temperature in San Francisco now? How about tomorrow?"},
    ]

    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'not_required'
    }

    # Prepare the data to be sent
    # model: The model defined by the user when setting up RKLLM-Server; this has no effect here
    # messages: The user's input question, which RKLLM-Server will use as input and return the model's reply; multiple questions can be added to messages
    # stream: Whether to enable streaming generate, should be False
    data = {
        "model": 'your_model_deploy_with_RKLLM_Server',
        "messages": messages,
        "stream": False,
        "enable_thinking": False,
        "tools": TOOLS
    }

    # Send a POST request
    responses = session.post(server_url, json=data, headers=headers, stream=False, verify=False)

    # Parse the response
    if responses.status_code == 200:
        print("Q:", data["messages"][-1]["content"], '\n')
        
        server_answer = json.loads(responses.text)["choices"][-1]["message"]["content"]
        matches = re.findall(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", ''.join(server_answer), re.DOTALL)
        print("server_answer:", server_answer, '\n')
        
        result = [json.loads(match) for match in matches]
        for function_call in result:
            messages.append({'role': 'assistant', 'content': '', 'function_call':function_call}) 

        tool_calls = [{'function': result[i]} for i in range(len(result))]
        function_call = []
        for tool_call in tool_calls:
            if fn_call := tool_call.get("function"):
                fn_name: str = fn_call["name"]
                fn_args: dict = fn_call["arguments"]
                fn_res: str = json.dumps(get_function_by_name(fn_name)(**fn_args))
                messages.append({'role': 'tool', 'name': fn_name, 'content':fn_res})

        print("messages:", messages, '\n')
        
    else:
        print("Error:", responses.text)
        exit()
       
    data = {
        "model": 'your_model_deploy_with_RKLLM_Server',
        "messages": messages,
        "stream": is_streaming,
        "enable_thinking": False,
        "tools": TOOLS
    }

    # Send a POST request
    responses = session.post(server_url, json=data, headers=headers, stream=is_streaming, verify=False)
    
    if not is_streaming:
        # Parse the response
        if responses.status_code == 200:
            print("A:", json.loads(responses.text)["choices"][-1]["message"]["content"])
        else:
            print("Error:", responses.text)
    else:
        if responses.status_code == 200:
            print("A:", end="")
            for line in responses.iter_lines():
                if line:
                    line = json.loads(line.decode('utf-8'))
                    if line["choices"][-1]["finish_reason"] != "stop":
                        print(line["choices"][-1]["delta"]["content"], end="")
                        sys.stdout.flush()
        else:
            print('Error:', responses.text)
            
    print('\n')


def main_demo3_multimodal(image_path=None, is_streaming=True, silent=False):
    """
    Demo for multimodal inference with image input.
    
    Args:
        image_path: Path to the image file. If None, will prompt user to input.
        is_streaming: Whether to use streaming mode.
        silent: If True, suppress print output and return result instead.
    
    Returns:
        If silent=True, returns the model's response text. Otherwise returns None.
    """
    if not silent:
        print("============================")
        print("RKLLM 多模态推理演示 (图片 + 文本)...")
        print("============================")
    
    # Get image path if not provided
    if image_path is None:
        image_path = input("\n*请输入图片文件路径: ")
    
    # Check if image exists
    if not os.path.exists(image_path):
        if not silent:
            print(f"错误: 找不到图片文件: {image_path}")
        return None
    

    # Read original image
    with open(image_path, 'rb') as f:
        image_data = f.read()
    
    # Encode image to base64
    try:
        image_base64 = base64.b64encode(image_data).decode('utf-8')
        
        # Add data URI prefix for proper format (optional, server handles both)
        # Detect image format from file extension
        ext = os.path.splitext(image_path)[1].lower()
        mime_type = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }.get(ext, 'image/jpeg')
        
        image_data_uri = f"data:{mime_type};base64,{image_base64}"
        
        if not silent:
            print(f"图片加载成功: {image_path}")
            print(f"图片大小: {len(image_data)} bytes")
        
    except Exception as e:
        if not silent:
            print(f"图片编码失败: {e}")
        return None
    
    # Get user's question about the image
    # user_question = input("\n*请输入关于这张图片的问题: ")
    # if not user_question:
    #     user_question = "请描述这张图片的内容。"  # Default question

    user_question = """根据当前监控画面判断以下3个问题（忽略OSD信息），答案只能是1或0。

    问题1：是否有车牌号出现在手机屏幕上？
    问题2：是否有车牌号出现在打印的纸张上？
    问题3：是否有人手持车牌？

    输出格式示例：0,1,0（只输出3个数字和2个逗号，不要其他内容）

    当前画面的答案："""
    
    # Prepare multimodal message
    # Format 1: Using image_url (OpenAI compatible format)
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": user_question
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_data_uri
                    }
                }
            ]
        }
    ]
    
    # Alternative Format 2: Using direct image field
    # messages = [
    #     {
    #         "role": "user",
    #         "content": [
    #             {
    #                 "type": "text",
    #                 "text": user_question
    #             },
    #             {
    #                 "type": "image",
    #                 "image": image_base64  # Can also use image_data_uri
    #             }
    #         ]
    #     }
    # ]
    
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'not_required'
    }
    
    data = {
        "model": 'your_model_deploy_with_RKLLM_Server',
        "messages": messages,
        "stream": is_streaming,
        "enable_thinking": False,
        "tools": None
    }
    
    if not silent:
        print("\nSending request to server...")
    
    # Send POST request
    try:
        responses = session.post(server_url, json=data, headers=headers, stream=is_streaming, verify=False)
        
        if not is_streaming:
            # Parse non-streaming response
            if responses.status_code == 200:
                result = json.loads(responses.text)["choices"][-1]["message"]["content"]
                if not silent:
                    print("\nQ:", user_question)
                    print("A:", result)
                else:
                    return result
            else:
                if not silent:
                    print("Error:", responses.text)
                return None
        else:
            # Parse streaming response
            if responses.status_code == 200:
                if not silent:
                    print("\nQ:", user_question)
                    print("A:", end="")
                result_text = ""
                for line in responses.iter_lines():
                    if line:
                        line = json.loads(line.decode('utf-8'))
                        if line["choices"][-1]["finish_reason"] != "stop":
                            content = line["choices"][-1]["delta"]["content"]
                            result_text += content
                            if not silent:
                                print(content, end="")
                                sys.stdout.flush()
                if not silent:
                    print()  # New line after streaming
                else:
                    return result_text
            else:
                if not silent:
                    print('Error:', responses.text)
                return None
                
    except Exception as e:
        if not silent:
            print(f"Error sending request: {e}")
        return None
    
    if not silent:
        print('\n')

def draw_text_on_image(image_path, text, output_path):
    """
    在图片右上角绘制文本
    
    Args:
        image_path: 输入图片路径
        text: 要绘制的文本
        output_path: 输出图片路径
    """
    try:
        # 打开图片并转换为RGBA模式以支持透明度
        img = Image.open(image_path)
        if img.mode != 'RGBA':
            img = img.convert('RGBA')
        
        # 获取图片尺寸
        img_width, img_height = img.size
        
        # 根据图片高度动态计算字体大小（使用图片高度的8%作为字体大小）
        font_size = int(img_height * 0.08)
        # 确保字体大小至少为100，最大为800
        font_size = max(100, min(font_size, 800))
        
        print(f"[调试] 图片尺寸: {img_width}x{img_height}, 计算字体大小: {font_size}px")
        
        # 尝试使用中文字体，如果失败则使用默认字体
        font = None
        font_loaded = False
        
        # 尝试多个可能的字体路径
        font_paths = [
            "/System/Library/Fonts/PingFang.ttc",
            "/System/Library/Fonts/Supplemental/Arial.ttf",
            "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        ]
        
        for font_path in font_paths:
            try:
                if os.path.exists(font_path):
                    font = ImageFont.truetype(font_path, font_size)
                    print(f"[调试] 成功加载字体: {font_path}, 字体大小: {font_size}px")
                    font_loaded = True
                    break
            except Exception as e:
                print(f"[调试] 尝试加载字体 {font_path} 失败: {e}")
                continue
        
        if not font_loaded:
            print(f"[警告] 所有字体加载失败，使用默认字体（可能很小）")
            # 默认字体不支持size参数，我们需要提示用户
            raise Exception("无法加载TrueType字体，请检查系统字体路径")
        
        # 创建一个透明图层用于绘制
        overlay = Image.new('RGBA', img.size, (255, 255, 255, 0))
        draw = ImageDraw.Draw(overlay)
        
        # 描边宽度（根据字体大小动态调整，约为字体大小的3%）
        stroke_width = max(5, int(font_size * 0.03))
        
        # 计算文本边界框（使用新的API，考虑描边宽度）
        bbox = draw.textbbox((0, 0), text, font=font, stroke_width=stroke_width)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # 计算文本位置（右上角，留一些边距，根据图片尺寸动态调整）
        margin = max(50, int(img_height * 0.03))  # 边距为图片高度的3%
        x = img_width - text_width - margin
        y = margin
        
        print(f"[调试] 文本尺寸: {text_width}x{text_height}, 位置: ({x}, {y})")
        print(f"[调试] 描边宽度: {stroke_width}px, 边距: {margin}px")
        
        # 绘制半透明黑色背景（提高不透明度，内边距根据字体大小调整）
        padding = max(30, int(font_size * 0.15))  # 内边距为字体大小的15%
        background_bbox = [x - padding, y - padding, x + text_width + padding, y + text_height + padding]
        draw.rectangle(background_bbox, fill=(0, 0, 0, 230))  # 提高背景不透明度到230
        
        # 绘制文本描边（黑色）增加字体可读性
        draw.text((x, y), text, font=font, fill=(255, 255, 255, 255), stroke_width=stroke_width, stroke_fill=(0, 0, 0, 255))
        
        print(f"[调试] 文本已绘制，内容: '{text}'")
        
        # 合并图层
        img = Image.alpha_composite(img, overlay)
        
        # 转换回RGB模式并保存
        if img.mode == 'RGBA':
            img = img.convert('RGB')
        
        img.save(output_path, quality=95)
        print(f"[调试] 图片已成功保存到: {output_path}")
        return True
        
    except Exception as e:
        print(f"[错误] 绘制文本失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def batch_process_folders(folder_paths, output_base_dir=None, is_streaming=False):
    """
    批量处理多个文件夹中的图片
    
    Args:
        folder_paths: 文件夹路径列表
        output_base_dir: 输出基础目录，如果为None则在原文件夹中创建output子文件夹
        is_streaming: 是否使用流式模式
    """
    print("="*60)
    print("开始批量处理图片...")
    print("="*60)
    
    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}
    
    total_processed = 0
    total_success = 0
    total_failed = 0
    
    for folder_path in folder_paths:
        folder_path = Path(folder_path)
        
        if not folder_path.exists():
            print(f"\n警告: 文件夹不存在: {folder_path}")
            continue
        
        print(f"\n处理文件夹: {folder_path}")
        print("-" * 60)
        
        # 创建输出目录
        if output_base_dir:
            output_dir = Path(output_base_dir) / folder_path.name
        else:
            output_dir = folder_path / "output"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"输出目录: {output_dir}")
        
        # 递归获取所有图片文件（包括子文件夹）
        image_files = []
        for ext in image_extensions:
            image_files.extend(folder_path.rglob(f"*{ext}"))
            image_files.extend(folder_path.rglob(f"*{ext.upper()}"))
        
        print(f"初始找到 {len(image_files)} 个图片文件")
        
        # 过滤掉我们自己创建的output输出目录中的文件（避免重复处理）
        # 使用绝对路径比较，只过滤output_dir下的文件，而不是所有包含'output'字符串的路径
        output_dir_abs = str(output_dir.resolve())
        print(f"输出目录绝对路径: {output_dir_abs}")
        
        before_filter = len(image_files)
        image_files = [f for f in image_files if not str(f.resolve()).startswith(output_dir_abs)]
        after_filter = len(image_files)
        
        if before_filter != after_filter:
            print(f"过滤掉 {before_filter - after_filter} 个输出目录中的文件")
        
        image_files = sorted(set(image_files))  # 去重并排序
        
        if not image_files:
            print(f"警告: 在 {folder_path} 中未找到图片文件")
            print(f"请检查:")
            print(f"  1. 目录中是否有图片文件（支持的格式: {', '.join(image_extensions)}）")
            print(f"  2. 图片是否都在output子目录中（会被过滤）")
            continue
        
        print(f"找到 {len(image_files)} 张图片")
        
        # 处理每张图片
        for idx, image_file in enumerate(image_files, 1):
            # 获取相对路径，用于显示和保存
            relative_path = image_file.relative_to(folder_path)
            print(f"\n[{idx}/{len(image_files)}] 处理: {relative_path}")
            
            try:
                # 调用大模型进行推理
                result = main_demo3_multimodal(
                    image_path=str(image_file),
                    is_streaming=is_streaming,
                    silent=True
                )
                
                if result:
                    print(f"  模型返回: {result}")
                    
                    # 生成输出文件路径，保持子文件夹结构
                    output_file = output_dir / relative_path
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    
                    # 获取图片尺寸用于显示调试信息
                    img_temp = Image.open(str(image_file))
                    img_w, img_h = img_temp.size
                    calculated_font_size = max(100, min(int(img_h * 0.08), 800))
                    print(f"  图片尺寸: {img_w}x{img_h}px, 字体大小: {calculated_font_size}px (图片高度的8%)")
                    
                    # 在图片上绘制结果
                    if draw_text_on_image(str(image_file), result, str(output_file)):
                        print(f"  ✓ 已保存: {output_file}")
                        total_success += 1
                    else:
                        print(f"  ✗ 绘制失败")
                        total_failed += 1
                else:
                    print(f"  ✗ 模型推理失败")
                    total_failed += 1
                
                total_processed += 1
                
            except Exception as e:
                print(f"  ✗ 处理失败: {e}")
                total_failed += 1
                total_processed += 1
    
    # 打印统计信息
    print("\n" + "="*60)
    print("批量处理完成！")
    print("="*60)
    print(f"总共处理: {total_processed} 张图片")
    print(f"成功: {total_success} 张")
    print(f"失败: {total_failed} 张")
    print("="*60)

def main_demo1(is_streaming=True):
    print("============================")
    print("Input your question in the terminal to start a conversation with the RKLLM model...")
    print("============================")
    # Enter a loop to continuously get user input and converse with the RKLLM model.
    while True:
        try:
            user_message = input("\n*Please enter your question:")
            if user_message == "exit":
                print("============================")
                print("The RKLLM Server is stopping......")
                print("============================")
                break
            else:
                # Set the request headers; in this case, the headers have no actual effect and are only used to simulate the OpenAI interface design.
                headers = {
                    'Content-Type': 'application/json',
                    'Authorization': 'not_required'
                }

                # Prepare the data to be sent
                # model: The model defined by the user when setting up RKLLM-Server; this has no effect here
                # messages: The user's input question, which RKLLM-Server will use as input and return the model's reply; multiple questions can be added to messages
                # stream: Whether to enable streaming conversation, similar to the OpenAI interface
                data = {
                    "model": 'your_model_deploy_with_RKLLM_Server',
                    "messages": [{"role": "user", "content": user_message}],
                    "stream": is_streaming,
                    "enable_thinking": False,
                    "tools": None
                }

                # Send a POST request
                responses = session.post(server_url, json=data, headers=headers, stream=is_streaming, verify=False)

                if not is_streaming:
                    # Parse the response
                    if responses.status_code == 200:
                        print("Q:", data["messages"][-1]["content"])
                        print("A:", json.loads(responses.text)["choices"][-1]["message"]["content"])
                    else:
                        print("Error:", responses.text)
                else:
                    if responses.status_code == 200:
                        print("Q:", data["messages"][-1]["content"])
                        print("A:", end="")
                        for line in responses.iter_lines():
                            if line:
                                line = json.loads(line.decode('utf-8'))
                                if line["choices"][-1]["finish_reason"] != "stop":
                                    print(line["choices"][-1]["delta"]["content"], end="")
                                    sys.stdout.flush()
                    else:
                        print('Error:', responses.text)
                        
        except KeyboardInterrupt:
            # Capture Ctrl-C signal to close the session
            session.close()

            print("\n")
            print("============================")
            print("The RKLLM Server is stopping......")
            print("============================")
            break
        
def list_system_fonts():
    """列出系统可用的字体文件"""
    print("="*60)
    print("系统字体检测...")
    print("="*60)
    
    font_dirs = [
        "/System/Library/Fonts/",
        "/System/Library/Fonts/Supplemental/",
        "/usr/share/fonts/truetype/",
        "/usr/share/fonts/",
    ]
    
    for font_dir in font_dirs:
        if os.path.exists(font_dir):
            print(f"\n检查目录: {font_dir}")
            for root, dirs, files in os.walk(font_dir):
                for file in files:
                    if file.endswith(('.ttf', '.ttc', '.otf')):
                        full_path = os.path.join(root, file)
                        print(f"  找到字体: {full_path}")
        else:
            print(f"\n目录不存在: {font_dir}")
    
    print("\n" + "="*60)

def test_draw_text(image_path, text="0,1,0"):
    """
    快速测试文本绘制功能（不调用大模型）
    
    Args:
        image_path: 测试图片路径
        text: 要绘制的测试文本
    """
    print("="*60)
    print("测试文本绘制功能...")
    print("="*60)
    
    if not os.path.exists(image_path):
        print(f"错误: 图片不存在: {image_path}")
        return
    
    # 生成测试输出文件
    output_path = os.path.join(os.path.dirname(image_path), "test_output.jpg")
    
    print(f"输入图片: {image_path}")
    print(f"测试文本: {text}")
    print(f"输出图片: {output_path}")
    print()
    
    # 绘制文本
    if draw_text_on_image(image_path, text, output_path):
        print(f"\n✓ 测试成功! 请查看输出图片: {output_path}")
    else:
        print(f"\n✗ 测试失败!")

if __name__ == '__main__':
    
    ## Demo0: 列出系统字体（调试用）
    # list_system_fonts()
    
    ## Demo1: RKLLM conversation
    # main_demo1(True)
    
    ## Demo2: RKLLM function-call
    # main_demo2(True)
    
    ## Demo3: RKLLM multimodal inference (image + text)
    # Example usage with specific image path:
    main_demo3_multimodal(image_path="/Users/shaoben/Downloads/ScreenShot_2025-10-24_092054_230.png", is_streaming=True)
    # Or let user input image path:
    # main_demo3_multimodal(is_streaming=True)
    
    ## Demo Test: 快速测试文本绘制（不调用大模型，快速验证字体大小是否合适）
    # 建议先运行这个测试，确认字体大小合适后再运行批量处理
    # test_draw_text("/Users/shaoben/Downloads/fakeplate/1759577964133192.168.2.248_浙JN21Z6_蓝色.jpg", "0,1,0")
    
    ## Demo4: Batch process folders
    # folder_paths = [
    #     "/Users/shaoben/output/"
    # ]
    # batch_process_folders(folder_paths, is_streaming=False)
    

