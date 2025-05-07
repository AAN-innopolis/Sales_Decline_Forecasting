"""
Определение сущности магазина для Feast.
"""

from feast import Entity

store_entity = Entity(
    name="store",
    description="Уникальный идентификатор магазина",
    join_keys=["store"],
) 