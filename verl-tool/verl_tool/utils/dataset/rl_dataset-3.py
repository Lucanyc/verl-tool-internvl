# 适配了InternVL3模型的rl_dataset.py
import io
import base64
import numpy as np
import regex as re
import time
import datasets
from verl.utils.dataset.rl_dataset import RLHFDataset
from pathlib import Path
from typing import List
from copy import deepcopy
from collections import defaultdict

# ==================== InternVL3 特殊 Token 定义 ====================
# InternVL3 使用以下特殊 token 来标记图像位置
INTERNVL3_VISION_START = '<|vision_start|>'  # Token ID: 151652
INTERNVL3_IMAGE_PAD = '<|image_pad|>'        # Token ID: 151655
INTERNVL3_VISION_END = '<|vision_end|>'      # Token ID: 151653
INTERNVL3_NUM_IMAGE_TOKENS = 256             # 每张图像对应的 token 数量

def get_internvl3_image_placeholder():
    """生成 InternVL3 的图像占位符序列"""
    return (
        INTERNVL3_VISION_START + 
        INTERNVL3_IMAGE_PAD * INTERNVL3_NUM_IMAGE_TOKENS + 
        INTERNVL3_VISION_END
    )

def encode_image(img_path: str) -> str:
    with open(img_path, "rb") as image_file:
        encoded_bytes = base64.b64encode(image_file.read())
        encoded_str = encoded_bytes.decode("utf-8")
        return encoded_str
    
def nested_copy(obj):
    """
    Recursively copy nested objects (lists, dicts, etc.) to avoid reference issues.
    """
    if isinstance(obj, dict):
        return {k: nested_copy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [nested_copy(item) for item in obj]
    elif hasattr(obj, 'copy'):
        return obj.copy()
    else:
        return obj
    
class RolloutMessagesMixin:
    """Mixin class to handle rollout messages in reinforcement learning datasets.

    This mixin provides methods to update and manage rollout messages, which are used
    to store the conversation history and interactions during the reinforcement learning process.
    """
    def __init__(self, messages: List[dict]):
        self.messages = messages if messages is not None else []
    
    def update_rollout_messages(self, new_message: dict) -> List[dict]:
        """Update the rollout messages with new messages."""
        messages = self.messages
        role = new_message['role']
        content_list = new_message['content']
        if isinstance(content_list, str):
            content_list = [{"type": "text", "text": content_list}]
        if isinstance(messages, np.ndarray):
            messages = messages.tolist()
        assert isinstance(content_list, list), f"content_list should be a list, but got {type(content_list)}"
        
        if messages[-1]['role'] != role:
            messages.append({'role': role, 'content': content_list})
        else:
            for content in content_list:
                if isinstance(content, dict) and content.get('type') == 'text' and messages[-1]['content'][-1].get('type') == 'text':
                    messages[-1]['content'][-1]['text'] += content['text']
                else:
                    messages[-1]['content'].append(content)
        return messages

    def tolist(self):
        """Convert the messages to a list format."""
        return self.messages.copy()
    
    def __copy__(self):
        """Create a shallow copy of the RolloutMessagesMixin instance."""
        return RolloutMessagesMixin(nested_copy(self.messages))
    
class VerlToolRLHFDataset(RLHFDataset):
    """A dataset class for reinforcement learning tasks in verl-tool.

    This class extends the base RLHFDataset class to provide additional functionality
    specific to verl-tool, such as custom data loading and processing methods.
    
    对于 InternVL3 模型，会自动将 <image> 占位符替换为：
    <|vision_start|><|image_pad|>×256<|vision_end|>
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._internvl3_detected = None  # 缓存检测结果
    
    def _is_internvl3(self) -> bool:
        """检测是否为 InternVL3 模型（带缓存）"""
        if self._internvl3_detected is not None:
            return self._internvl3_detected
        
        self._internvl3_detected = False
        
        if self.tokenizer is not None:
            try:
                # 方法1: 检查 <|im_end|> token (ID: 151645)
                token = self.tokenizer.convert_ids_to_tokens(151645)
                if token == '<|im_end|>':
                    # 方法2: 进一步确认 InternVL3 特有的 token
                    vision_start_id = self.tokenizer.convert_tokens_to_ids(INTERNVL3_VISION_START)
                    image_pad_id = self.tokenizer.convert_tokens_to_ids(INTERNVL3_IMAGE_PAD)
                    vision_end_id = self.tokenizer.convert_tokens_to_ids(INTERNVL3_VISION_END)
                    
                    if vision_start_id == 151652 and image_pad_id == 151655 and vision_end_id == 151653:
                        self._internvl3_detected = True
                        print(f"[VerlToolRLHFDataset] ✅ Detected InternVL3 model")
                        print(f"[VerlToolRLHFDataset]   - vision_start: {INTERNVL3_VISION_START} -> ID {vision_start_id}")
                        print(f"[VerlToolRLHFDataset]   - image_pad: {INTERNVL3_IMAGE_PAD} -> ID {image_pad_id}")
                        print(f"[VerlToolRLHFDataset]   - vision_end: {INTERNVL3_VISION_END} -> ID {vision_end_id}")
                    else:
                        # 可能是旧版 InternVL，打印警告
                        print(f"[VerlToolRLHFDataset] ⚠️ Detected InternVL but NOT InternVL3")
                        print(f"[VerlToolRLHFDataset]   - vision_start ID: {vision_start_id} (expected 151652)")
                        print(f"[VerlToolRLHFDataset]   - image_pad ID: {image_pad_id} (expected 151655)")
                        print(f"[VerlToolRLHFDataset]   - vision_end ID: {vision_end_id} (expected 151653)")
            except Exception as e:
                print(f"[VerlToolRLHFDataset] InternVL detection error: {e}")
        
        return self._internvl3_detected
    
    def _replace_image_placeholder_for_internvl3(self, content: str) -> str:
        """将 <image> 替换为 InternVL3 格式的占位符"""
        if '<image>' not in content:
            return content
        
        internvl3_placeholder = get_internvl3_image_placeholder()
        return content.replace('<image>', internvl3_placeholder)
    

    def __getitem__(self, item):
        """
        Note that we also return the raw_input_ids so that it can be combined with other chat template
        """
        row_dict: dict = self.dataframe[item]
        start = time.time()
        rollout_messages = self._build_rollout_messages(row_dict)
        result = super().__getitem__(item)
        result['rollout_messages'] = rollout_messages
        
        # [DEBUG] 检查 raw_prompt_ids 中是否包含视觉 token
        if self._is_internvl3() and 'raw_prompt_ids' in result:
            raw_ids = result['raw_prompt_ids']
            if hasattr(raw_ids, 'tolist'):
                raw_ids = raw_ids.tolist()
            print(f"[__getitem__ DEBUG] 151655 in raw_prompt_ids: {151655 in raw_ids}")
            print(f"[__getitem__ DEBUG] 151652 in raw_prompt_ids: {151652 in raw_ids}")
            print(f"[__getitem__ DEBUG] raw_prompt_ids[:30]: {raw_ids[:30]}")
        

        return result
    
    def _build_messages(self, example: dict):
        """Override to handle InternVL3 format"""
        
        if self._is_internvl3():
            # InternVL3 需要特殊处理 - 深拷贝 messages
            messages = deepcopy(example[self.prompt_key])
            
            # 对 user 消息进行 <image> 替换
            for message in messages:
                if message['role'] == 'user' and isinstance(message.get('content'), str):
                    if '<image>' in message['content']:
                        image_count = message['content'].count('<image>')
                        message['content'] = self._replace_image_placeholder_for_internvl3(message['content'])
                        # 只打印一次调试信息
                        if not hasattr(self, '_build_messages_logged'):
                            self._build_messages_logged = True
                            print(f"[_build_messages] Replaced {image_count} <image> -> InternVL3 format")
            
            return messages
        else:
            # 其他模型调用父类方法
            return super()._build_messages(example)
    
    def maybe_filter_out_long_prompts(self, dataframe: datasets.Dataset = None):
        # filter out too long prompts
        if self.filter_overlong_prompts:
            tokenizer = self.tokenizer
            processor = self.processor
            prompt_key = self.prompt_key
            image_key = self.image_key
            video_key = self.video_key

            if processor is not None:
                from verl.utils.dataset.vision_utils import process_image, process_video

                def doc2len(doc) -> int:
                    messages = self._build_messages(doc)
                    raw_prompt = self.processor.apply_chat_template(
                        messages, add_generation_prompt=True, tokenize=False
                    )
                    images = (
                        [process_image(image) for image in doc[image_key]] if image_key in doc else None
                    )
                    videos = (
                        [process_video(video) for video in doc[video_key]] if video_key in doc else None
                    )

                    return len(processor(text=[raw_prompt], images=images, videos=videos)["input_ids"][0])

            else:

                def doc2len(doc) -> int:
                    return len(tokenizer.apply_chat_template(doc[prompt_key], add_generation_prompt=True))

            dataframe = dataframe.filter(
                lambda doc: doc2len(doc) <= self.max_prompt_length,
                num_proc=self.num_workers,
                desc=f"Filtering prompts longer than {self.max_prompt_length} tokens",
            )

            print(f"filter dataset len: {len(dataframe)}")
        return dataframe
    
    def _build_rollout_messages(self, example: dict):
        messages = deepcopy(example[self.prompt_key])
        
        # 检测是否为 InternVL3
        is_internvl3 = self._is_internvl3()
        
        # InternVL3: 保持字符串格式，但需要替换 <image> 标签
        if is_internvl3:
            # 只处理 user 消息，不处理 system 消息
            for idx, message in enumerate(messages):
                if message['role'] == 'user' and isinstance(message.get('content'), str):
                    original_content = message['content']
                    if '<image>' in original_content:
                        # 统计图片数量
                        image_count = original_content.count('<image>')
                        
                        # 替换每个 <image> 为 InternVL3 格式
                        message['content'] = self._replace_image_placeholder_for_internvl3(original_content)
                        
                        # 调试日志（只在首次打印）
                        if not hasattr(self, '_rollout_logged'):
                            self._rollout_logged = True
                            print(f"[_build_rollout_messages] ✅ Replaced {image_count} <image> tags")
                            print(f"[_build_rollout_messages] Format: <|vision_start|><|image_pad|>×{INTERNVL3_NUM_IMAGE_TOKENS}<|vision_end|>")
                            # 显示部分替换后的内容
                            preview = message['content'][:300].replace('\n', '\\n')
                            print(f"[_build_rollout_messages] Preview: {preview}...")
            
            return RolloutMessagesMixin(messages)
        
        # ==================== 非 InternVL3 模型的处理逻辑 ====================
        # 其他模型（Qwen2.5-VL等）的 content 转换逻辑
        if self.image_key in example or self.video_key in example:
            for message in messages:
                content = message["content"]
                content_list = []
                try:
                    segments = re.split("(<image>|<video>)", content)
                except Exception as e:
                    raise ValueError(f"Error splitting content: {content}") from e
                segments = [item for item in segments if item != ""]
                segment_idx = defaultdict(int)
                for segment in segments:
                    if segment == "<image>":
                        content_list.append({"type": "image", "image": example[self.image_key][segment_idx[segment]]["image"]})
                        segment_idx[segment] += 1
                    elif segment == "<video>":
                        content_list.append({"type": "video", "video": example[self.video_key][segment_idx[segment]]["video"]})
                        segment_idx[segment] += 1
                    else:
                        content_list.append({"type": "text", "text": segment})

                message["content"] = content_list

        # 对于非 InternVL3 的处理器
        if self.processor is not None and not is_internvl3:
            # multi-modal inputs conversion for non-InternVL models
            from verl_tool.llm_agent.vision_utils import encode_image_url, encode_video_url
            for i, message in enumerate(messages):
                if isinstance(message['content'], list):
                    for j in range(len(message['content'])):
                        content = message['content'][j]
                        if content['type'] == 'image':
                            message['content'][j] = {
                                "type": "image_url",
                                "image_url": {
                                    "url": encode_image_url(content['image']),
                                }
                            }
                            assert Path(content['image']).exists(), f"Image file {content['image']} does not exist."
                        elif content['type'] == 'video':
                            message['content'][j] = {
                                "type": "video_url",
                                "video_url": {
                                    "url": encode_video_url(content['video']),
                                }
                            }
                            assert Path(content['video']).exists(), f"Video file {content['video']} does not exist."
                        elif content['type'] == 'text':
                            message['content'][j] = {
                                "type": "text",
                                "text": content['text']
                            }
                        else:
                            raise ValueError(f"Unknown content element type: {content['type']}")
                elif isinstance(message['content'], str):
                    message['content'] = [{"type": "text", "text": message['content']}]
                else:
                    raise ValueError(f"Unknown content type: {type(message['content'])}")
                    
        return RolloutMessagesMixin(messages)


