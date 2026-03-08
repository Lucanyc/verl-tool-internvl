#!/usr/bin/env python3
"""
测试InternVL3官方流程的工具调用
验证消息流: user → assistant (tool call) → function (tool result) → assistant (reasoning + answer)
"""

import asyncio
import aiohttp
import json
import os
from transformers import AutoTokenizer

class TestInternVL3ToolFlow:
    def __init__(self):
        self.router_url = "http://localhost:5556/get_observation"
        self.tokenizer = AutoTokenizer.from_pretrained(
            "/host_home/SFT-Internvl-official/outputs/fixed_training-with-think",
            trust_remote_code=True
        )
        
        # 使用实际存在的测试图片路径
        self.test_images = {
            "chart": "/host_home/dataset/ChartQA/ChartQA Dataset/test/png/41699051005347.png",
            "geometry": "/host_home/dataset/geometry3k/test_data/test/2401/img_diagram.png",
            "diagram": "/host_home/dataset/geometry3k/test_data/test/2402/img_diagram.png",
            # 备用路径
            "chartqa_alt": "/host_home/VlmGym/data/chartqa/train/png/two_col_63384.png",
            "iconqa": "/host_home/VlmGym/data/IconQA/iconqa_data/iconqa/train/choose_txt/10/image.png"
        }
    
    def get_test_image(self, type="chart"):
        """获取测试图片路径"""
        path = self.test_images.get(type, self.test_images["chart"])
        if os.path.exists(path):
            return path
        else:
            # 如果路径不存在，尝试找一个存在的
            print(f"   ⚠️ 首选路径不存在: {path}")
            for img_type, p in self.test_images.items():
                if os.path.exists(p):
                    print(f"   ✅ 使用备用路径: {p}")
                    return p
            raise FileNotFoundError("No test image found in any configured path")
    
    async def test_complete_flow(self):
        """测试完整的InternVL3工具调用流程"""
        print("="*70)
        print("🚀 测试InternVL3官方流程 (function role)")
        print("="*70)
        
        # 测试案例 - 添加了 qid 和 is_last_step
        test_cases = [
            {
                "name": "EasyOCR文字识别",
                "messages": [
                    {"role": "user", "content": "Extract text from this image"},
                    {"role": "assistant", "content": '<tool_call>{"tool": "easyocr", "task": "extract"}</tool_call>'}
                ],
                "router_request": {
                    "trajectory_ids": ["test_easyocr"],
                    "actions": ['<tool_call>{"tool": "easyocr", "task": "extract"}</tool_call>'],
                    "finish": [False],
                    "is_last_step": [False],  # ✅ 添加
                    "extra_fields": [{
                        "images": [self.get_test_image("chart")],
                        "qid": "test_easyocr_001"  # ✅ 添加
                    }]
                }
            },
            {
                "name": "ChartMoE图表分析（带choices）",
                "messages": [
                    {"role": "user", "content": "How many data points are shown? A. 3 B. 4 C. 5 D. 6"},
                    {"role": "assistant", "content": '<tool_call>{"tool": "chartmoe", "task": "analyze"}</tool_call>'}
                ],
                "router_request": {
                    "trajectory_ids": ["test_chartmoe"],
                    "actions": ['<tool_call>{"tool": "chartmoe", "task": "analyze"}</tool_call>'],
                    "finish": [False],
                    "is_last_step": [False],  # ✅ 添加
                    "extra_fields": [{
                        "images": [self.get_test_image("chart")],
                        "qid": "test_chartmoe_001",  # ✅ 添加
                        "question": "How many data points are shown?",
                        "choices": ["3", "4", "5", "6"]
                    }]
                }
            },
            {
                "name": "MultiMath几何计算（带choices）",
                "messages": [
                    {"role": "user", "content": "Find the value of x in the triangle. A. 30 B. 60 C. 120 D. 240"},
                    {"role": "assistant", "content": '<tool_call>{"tool": "multimath", "task": "solve"}</tool_call>'}
                ],
                "router_request": {
                    "trajectory_ids": ["test_multimath"],
                    "actions": ['<tool_call>{"tool": "multimath", "task": "solve"}</tool_call>'],
                    "finish": [False],
                    "is_last_step": [False],  # ✅ 添加
                    "extra_fields": [{
                        "images": [self.get_test_image("geometry")],
                        "qid": "test_multimath_001",  # ✅ 添加
                        "question": "Find the value of x in the triangle",
                        "choices": ["30", "60", "120", "240"]  # MultiMath 也支持 choices
                    }]
                }
            },
            {
                "name": "G-LLaVA几何分析（带choices）",
                "messages": [
                    {"role": "user", "content": "What is the angle measurement? A. 30° B. 45° C. 60° D. 90°"},
                    {"role": "assistant", "content": '<tool_call>{"tool": "gllava", "task": "solve"}</tool_call>'}
                ],
                "router_request": {
                    "trajectory_ids": ["test_gllava"],
                    "actions": ['<tool_call>{"tool": "gllava", "task": "solve"}</tool_call>'],
                    "finish": [False],
                    "is_last_step": [False],  # ✅ 添加
                    "extra_fields": [{
                        "images": [self.get_test_image("geometry")],
                        "qid": "test_gllava_001",  # ✅ 添加
                        "question": "What is the angle measurement?",
                        "choices": ["30°", "45°", "60°", "90°"]
                    }]
                }
            },
            {
                "name": "DiagramFormalizer图表形式化",
                "messages": [
                    {"role": "user", "content": "Formalize this diagram"},
                    {"role": "assistant", "content": '<tool_call>{"tool": "diagramformalizer", "task": "analyze"}</tool_call>'}
                ],
                "router_request": {
                    "trajectory_ids": ["test_diagramformalizer"],
                    "actions": ['<tool_call>{"tool": "diagramformalizer", "task": "analyze"}</tool_call>'],
                    "finish": [False],
                    "is_last_step": [False],
                    "extra_fields": [{
                        "images": [self.get_test_image("diagram")],
                        "qid": "test_diagramformalizer_001"
                    }]
                }
            }
        ]
        
        success_count = 0
        total_count = len(test_cases)
        
        for test in test_cases:
            print(f"\n{'='*70}")
            print(f"📝 测试: {test['name']}")
            print(f"{'='*70}")
            
            # 1. 显示消息流
            messages = test["messages"].copy()
            print("\n1️⃣ 初始消息:")
            for msg in messages:
                content_preview = msg['content'][:100] + "..." if len(msg['content']) > 100 else msg['content']
                print(f"   {msg['role']}: {content_preview}")
            
            # 2. 显示请求详情
            print("\n2️⃣ Router请求:")
            ef = test["router_request"]["extra_fields"][0]
            image_path = ef['images'][0]
            print(f"   图片: .../{'/'.join(image_path.split('/')[-3:])}")
            print(f"   QID: {ef.get('qid', 'N/A')}")
            if 'question' in ef:
                print(f"   问题: {ef['question']}")
            if 'choices' in ef:
                print(f"   选项: {ef['choices']}")
            
            # 3. 调用工具
            print("\n3️⃣ 调用工具...")
            tool_result = await self.call_tool_router(test["router_request"])
            
            if tool_result:
                success_count += 1
                
                # 4. 添加function消息
                messages.append({
                    "role": "function",
                    "content": tool_result
                })
                print(f"\n4️⃣ 工具返回 (function role):")
                result_preview = tool_result[:400] + "..." if len(tool_result) > 400 else tool_result
                print(f"   {result_preview}")
                
                # 5. 模拟assistant最终回复（带推理过程）
                final_answer = self.generate_final_answer(tool_result, test["name"])
                messages.append({
                    "role": "assistant", 
                    "content": final_answer
                })
                print(f"\n5️⃣ Assistant最终回复:")
                answer_preview = final_answer[:300] + "..." if len(final_answer) > 300 else final_answer
                print(f"   {answer_preview}")
                
                # 6. 验证消息格式
                print("\n6️⃣ 验证消息格式:")
                if self.verify_message_format(messages):
                    print("   ✅ 消息流验证通过!")
                else:
                    print("   ⚠️ 消息流验证存在问题")
                
            else:
                print("   ❌ 工具调用失败")
        
        # 总结
        print(f"\n{'='*70}")
        print(f"📊 测试总结:")
        print(f"   成功: {success_count}/{total_count}")
        print(f"   失败: {total_count - success_count}/{total_count}")
        print(f"{'='*70}")
    
    def generate_final_answer(self, tool_result, test_name):
        """
        根据工具结果生成最终答案
        遵循 InternVL3 格式: <think>推理过程</think> <answer>答案</answer>
        """
        # 提取工具返回的关键信息
        if "EasyOCR" in tool_result:
            extracted_text = tool_result.split("Result:")[-1].strip()[:200]
            reasoning = f"The EasyOCR tool has successfully extracted text from the image. The detected text includes various elements that appear in the image."
            answer = f"The extracted text is: {extracted_text}"
            
        elif "ChartMoE" in tool_result:
            analysis = tool_result.split("Result:")[-1].strip()[:200]
            reasoning = f"Based on the ChartMoE analysis of the chart, I can identify the data points and their values. The chart structure has been analyzed and the relevant information extracted."
            answer = f"According to the chart analysis: {analysis}"
            
        elif "MultiMath" in tool_result:
            solution = tool_result.split("Solution:")[-1].strip()[:200]
            reasoning = f"Using the MultiMath solver, I can work through the geometric problem step by step. The mathematical relationships in the triangle help determine the value."
            answer = f"The solution is: {solution}"
            
        elif "G-LLaVA" in tool_result or "Geometric" in tool_result:
            geo_analysis = tool_result.split("Analysis:")[-1].strip()[:200]
            reasoning = f"The G-LLaVA geometric analysis tool has examined the figure and determined the angle measurements based on the geometric properties."
            answer = f"The geometric analysis shows: {geo_analysis}"
            
        elif "DiagramFormalizer" in tool_result:
            formalized = tool_result.split("Analysis:")[-1].strip()[:200]
            reasoning = f"The DiagramFormalizer has processed the diagram and extracted its formal representation, including the structural elements and relationships."
            answer = f"The formalized diagram structure is: {formalized}"
            
        else:
            reasoning = f"Based on the tool output, I can provide the following information."
            answer = f"Result: {tool_result[:150]}"
        
        # 组合成 InternVL3 格式
        return f"<think>{reasoning}</think><answer>{answer}</answer>"
    
    async def call_tool_router(self, request_data):
        """调用tool router"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.router_url,
                    json=request_data,
                    timeout=aiohttp.ClientTimeout(total=120)  # 增加到2分钟
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        obs = result["observations"][0]
                        valid = result["valids"][0]
                        
                        if valid and not obs.startswith("Error"):
                            print(f"   ✅ 工具调用成功")
                            return obs
                        else:
                            print(f"   ⚠️ 工具返回错误: {obs[:300]}")
                            return None
                    else:
                        print(f"   ❌ HTTP错误: {response.status}")
                        text = await response.text()
                        print(f"   响应: {text[:300]}")
                        return None
                        
        except asyncio.TimeoutError:
            print(f"   ❌ 请求超时 (120秒)")
            return None
        except aiohttp.ClientError as e:
            print(f"   ❌ 网络错误: {e}")
            return None
        except Exception as e:
            print(f"   ❌ 异常: {type(e).__name__}: {e}")
            return None
    
    def verify_message_format(self, messages):
        """验证消息格式符合InternVL3标准"""
        try:
            # 使用tokenizer应用聊天模板
            formatted = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=False
            )
            
            # 检查关键元素
            checks = [
                ("<|im_start|>user" in formatted, "✓ user消息"),
                ("<|im_start|>assistant" in formatted, "✓ assistant消息"),
                ("<|im_start|>function" in formatted, "✓ function消息 (关键!)"),
                (formatted.count("<|im_start|>assistant") >= 2, "✓ 至少2个assistant消息"),
                ("</tool_call>" in formatted, "✓ tool_call标签"),
                ("<think>" in formatted, "✓ 推理标签 <think>"),
                ("<answer>" in formatted, "✓ 答案标签 <answer>"),
            ]
            
            all_passed = True
            print("")
            for passed, desc in checks:
                status = "   ✅" if passed else "   ❌"
                print(f"{status} {desc}")
                if not passed:
                    all_passed = False
            
            if all_passed:
                print("\n   ✨ 消息流完全符合InternVL3标准!")
                
                # 显示格式化后的消息结构
                print("\n   📋 消息结构预览:")
                lines = formatted.split('\n')
                for i, line in enumerate(lines):
                    if '<|im_start|>' in line:
                        role = line.replace('<|im_start|>', '').strip()
                        # 查找内容预览
                        content_preview = ""
                        if i + 1 < len(lines):
                            content_preview = lines[i + 1][:50]
                            if len(lines[i + 1]) > 50:
                                content_preview += "..."
                        print(f"      → {role}: {content_preview}")
            else:
                print("\n   ⚠️ 部分检查未通过")
                print(f"\n   调试信息 - 格式化消息前100字符:")
                print(f"   {formatted[:100]}...")
            
            return all_passed
            
        except Exception as e:
            print(f"   ❌ 验证失败: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return False

async def check_services():
    """检查必要的服务是否在运行"""
    services = {
        "Tool Router": "http://localhost:5556/health",
        "EasyOCR": "http://localhost:5758/health",
        "ChartMoE": "http://localhost:5658/health",
        "MultiMath": "http://localhost:5582/health",
        "G-LLaVA": "http://localhost:7690/health",
        "DiagramFormalizer": "http://localhost:6866/health",
    }
    
    print("🔍 检查服务状态...")
    print("="*70)
    
    all_running = True
    async with aiohttp.ClientSession() as session:
        for name, url in services.items():
            try:
                async with session.get(url, timeout=aiohttp.ClientTimeout(total=2)) as response:
                    if response.status == 200:
                        print(f"   ✅ {name}: 运行中")
                    else:
                        print(f"   ⚠️ {name}: 响应异常 ({response.status})")
                        all_running = False
            except asyncio.TimeoutError:
                print(f"   ❌ {name}: 超时")
                all_running = False
            except aiohttp.ClientError:
                print(f"   ❌ {name}: 无法连接")
                all_running = False
            except Exception as e:
                print(f"   ❌ {name}: {type(e).__name__}")
                all_running = False
    
    print("="*70)
    
    if not all_running:
        print("\n⚠️ 警告: 部分服务未运行，某些测试可能失败")
        print("提示: 使用 ./start_tool_servers.sh 启动所有服务")
    else:
        print("\n✅ 所有服务运行正常!")
    
    print("")
    return all_running

async def main():
    print("="*70)
    print("🔍 InternVL3工具调用流程测试")
    print("验证消息流: user → assistant (tool) → function (result) → assistant (reasoning)")
    print("="*70)
    print("")
    
    # 检查服务
    await check_services()
    
    # 运行测试
    tester = TestInternVL3ToolFlow()
    await tester.test_complete_flow()
    
    print("\n" + "="*70)
    print("✅ 所有测试完成")
    print("="*70)

if __name__ == "__main__":
    asyncio.run(main())