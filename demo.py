import logging

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s: %(message)s",
                    level=logging.DEBUG,
                    filename='./new.txt',
                    filemode='w')
logger = logging.getLogger("user")
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler = logging.FileHandler('./log_demo.txt', 'w')
handler.setFormatter(formatter)
handler.setLevel(level=logging.DEBUG)
logger.addHandler(handler)

logger.info("asasdasdd")
logger.debug("123123asdfsdasdds")
logger.warning("123123asdasdsdafsds")
