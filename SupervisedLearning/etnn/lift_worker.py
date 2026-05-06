# lift_worker.py
import sys
import os
import pickle
import torch

# ✨ 核心修复：强制将当前运行目录加入环境路径，确保子进程能找到 etnn
current_dir = os.getcwd()
if current_dir not in sys.path:
    sys.path.append(current_dir)

def work():
    try:
        # 从标准输入读取数据
        raw_input = sys.stdin.buffer.read()
        if not raw_input:
            sys.exit(1)
            
        input_data = pickle.loads(raw_input)
        transform = input_data['transform']
        base_data = input_data['base_data']

        # 执行 Lifting
        cc_data = transform(base_data)
        
        # 返回结果
        sys.stdout.buffer.write(pickle.dumps(cc_data))
        sys.exit(0)
    except Exception as e:
        # 如果是 Python 层面的报错，将其写到 stderr 方便主进程捕获
        sys.stderr.write(str(e))
        sys.exit(1) # 非正常退出

if __name__ == "__main__":
    work()