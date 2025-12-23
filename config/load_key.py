def load_key():
    """
    加载 DashScope API Key
    
    优先级：
    1. 环境变量 DASHSCOPE_API_KEY（云服务器推荐）
    2. config/Key.json 文件（本地开发推荐）
    3. 用户交互式输入（交互式终端）
    """
    import os
    import getpass
    import json
    import dashscope
    from pathlib import Path
    
    # 优先使用环境变量（云服务器友好）
    api_key = os.getenv('DASHSCOPE_API_KEY')
    if api_key:
        api_key = api_key.strip()
        os.environ['DASHSCOPE_API_KEY'] = api_key
        dashscope.api_key = api_key
        print(f"✅ 从环境变量加载 API Key: {api_key[:5]}{'*' * 5}")
        return
    
    # 回退到文件加载（本地开发友好）
    # 使用 Path(__file__) 确保路径相对于脚本位置，而不是工作目录
    script_dir = Path(__file__).parent
    file_name = script_dir / 'Key.json'
    
    if file_name.exists():
        with open(file_name, 'r') as file:
            Key = json.load(file)
        if "DASHSCOPE_API_KEY" in Key:
            api_key = Key["DASHSCOPE_API_KEY"].strip()
            os.environ['DASHSCOPE_API_KEY'] = api_key
            dashscope.api_key = api_key
            print(f"✅ 从配置文件加载 API Key: {api_key[:5]}{'*' * 5}")
            return
    
    # 最后回退到交互式输入（需要终端交互）
    try:
        DASHSCOPE_API_KEY = getpass.getpass("未找到存放Key的文件，请输入你的api_key:").strip()
        Key = {
            "DASHSCOPE_API_KEY": DASHSCOPE_API_KEY
        }
        with open(file_name, 'w') as json_file:
            json.dump(Key, json_file, indent=4)
        os.environ['DASHSCOPE_API_KEY'] = DASHSCOPE_API_KEY
        dashscope.api_key = DASHSCOPE_API_KEY
        print(f"✅ API Key 已保存到: {file_name}")
    except (EOFError, KeyboardInterrupt):
        print("\n❌ 未配置 API Key，请通过以下方式之一配置：")
        print("  1. 环境变量: export DASHSCOPE_API_KEY='your-key'")
        print(f"  2. 配置文件: 创建 {file_name} 并添加 DASHSCOPE_API_KEY")
        raise RuntimeError("API Key 未配置")

if __name__ == '__main__':
    load_key()
    import os
    print(os.environ['DASHSCOPE_API_KEY'])