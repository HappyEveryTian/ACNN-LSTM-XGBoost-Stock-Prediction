import logging

# 创建一个名为 __name__ 的 logger
logger = logging.getLogger(__name__)
# 设置日志级别
logger.setLevel(logging.DEBUG)

# 创建一个 FileHandler，将日志写入文件
file_handler = logging.FileHandler('../model_metrics.log', encoding='utf-8')
# 创建一个 Formatter 对象，设置日志格式
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# 将 FileHandler 添加到 logger
logger.addHandler(file_handler)