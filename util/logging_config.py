import logging

# ����һ����Ϊ __name__ �� logger
logger = logging.getLogger(__name__)
# ������־����
logger.setLevel(logging.DEBUG)

# ����һ�� FileHandler������־д���ļ�
file_handler = logging.FileHandler('../model_metrics.log', encoding='utf-8')
# ����һ�� Formatter ����������־��ʽ
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# �� FileHandler ��ӵ� logger
logger.addHandler(file_handler)