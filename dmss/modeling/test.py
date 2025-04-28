from train import Config

from dmss.models import PolypModel

# --------Algotithm-----------
# 1. Инициализировать модель
# 2. Загрузить веса модели
# 3. загрузить тестовый датасет
# 3. Прогнать изображение через модель
# 4. Получить предсказание и вернуть его
# 5. Посчитать метрики
# 6. Вузуализировать предсказания


conf = Config()
model = PolypModel(
    arch=conf.arch,
    encoder_name=conf.encoder_name,
    in_channels=conf.in_channels,
    out_classes=conf.out_classes,
    device=conf.device,
)
