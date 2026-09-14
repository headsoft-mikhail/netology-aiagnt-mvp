import enum
import typing

import pydantic

from ai_agent import contracts


class ProductCategory(enum.StrEnum):
    ROUTER = "router"
    MESH_SYSTEM = "mesh_system"
    NETWORK_SWITCH = "network_switch"
    WIFI_ADAPTER = "wifi_adapter"
    ACCESS_POINT = "access_point"


class ConnectionType(enum.StrEnum):
    USB = "usb"
    PCI_E = "pci_e"


class Product(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid", str_strip_whitespace=True)

    sku: str = pydantic.Field(min_length=1)
    name: str = pydantic.Field(min_length=1)
    category: ProductCategory
    brand: str = pydantic.Field(min_length=1)
    price_rub: int = pydantic.Field(gt=0)
    in_stock: bool
    wifi_generation: int | None = pydantic.Field(default=None, ge=4, le=7)
    max_wireless_speed_mbps: int | None = pydantic.Field(default=None, gt=0)
    wan_speed_mbps: int | None = pydantic.Field(default=None, gt=0)
    lan_ports: int | None = pydantic.Field(default=None, ge=0)
    mesh_support: bool | None = None
    nodes: int | None = pydantic.Field(default=None, gt=0)
    coverage_sqm: int | None = pydantic.Field(default=None, gt=0)
    port_count: int | None = pydantic.Field(default=None, gt=0)
    port_speed_mbps: int | None = pydantic.Field(default=None, gt=0)
    managed: bool | None = None
    poe: bool | None = None
    connection_type: ConnectionType | None = None


class BaseProductFilters(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    max_price_rub: int | None = pydantic.Field(default=None, gt=0)
    brands: list[str] = pydantic.Field(default_factory=list)
    in_stock: bool | None = True


class RouterFilters(BaseProductFilters):
    category: typing.Literal[ProductCategory.ROUTER]
    min_wifi_generation: int | None = pydantic.Field(default=None, ge=4, le=7)
    min_wireless_speed_mbps: int | None = pydantic.Field(default=None, gt=0)
    min_wan_speed_mbps: int | None = pydantic.Field(default=None, gt=0)
    min_lan_ports: int | None = pydantic.Field(default=None, ge=0)
    mesh_support: bool | None = None


class MeshSystemFilters(BaseProductFilters):
    category: typing.Literal[ProductCategory.MESH_SYSTEM]
    min_wifi_generation: int | None = pydantic.Field(default=None, ge=4, le=7)
    min_wan_speed_mbps: int | None = pydantic.Field(default=None, gt=0)
    min_nodes: int | None = pydantic.Field(default=None, gt=0)
    min_coverage_sqm: int | None = pydantic.Field(default=None, gt=0)


class NetworkSwitchFilters(BaseProductFilters):
    category: typing.Literal[ProductCategory.NETWORK_SWITCH]
    min_port_count: int | None = pydantic.Field(default=None, gt=0)
    min_port_speed_mbps: int | None = pydantic.Field(default=None, gt=0)
    managed: bool | None = None
    poe: bool | None = None


class WifiAdapterFilters(BaseProductFilters):
    category: typing.Literal[ProductCategory.WIFI_ADAPTER]
    min_wifi_generation: int | None = pydantic.Field(default=None, ge=4, le=7)
    min_wireless_speed_mbps: int | None = pydantic.Field(default=None, gt=0)
    connection_type: ConnectionType | None = None


class AccessPointFilters(BaseProductFilters):
    category: typing.Literal[ProductCategory.ACCESS_POINT]
    min_wifi_generation: int | None = pydantic.Field(default=None, ge=4, le=7)
    min_wireless_speed_mbps: int | None = pydantic.Field(default=None, gt=0)
    poe: bool | None = None


type ProductFilters = typing.Annotated[
    RouterFilters | MeshSystemFilters | NetworkSwitchFilters | WifiAdapterFilters | AccessPointFilters,
    pydantic.Field(discriminator="category"),
]


class ProductSearchInput(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    filters: ProductFilters
    limit: int = pydantic.Field(default=10, ge=1, le=50)


class ProductSearchResult(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")

    status: contracts.ToolStatus
    products: list[Product] = pydantic.Field(default_factory=list)
    total: int = pydantic.Field(default=0, ge=0)
    error: contracts.ToolError | None = None
