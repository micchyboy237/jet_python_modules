from typing import Any, TypedDict, List, Dict, Optional
from pydantic import BaseModel

# Auth


class AuthRequest(TypedDict):
    host: str
    port: int
    username: str
    password: str
    isRouted: bool
    isEncrypted: bool
    wsPort: int
    databaseName: str
    appId: str


class AuthInfo(TypedDict):
    protocol: str
    host: str
    port: int
    wsPort: int
    uri: str
    isEncrypted: bool
    isRouted: bool
    version: str
    databaseName: str
    serverType: str
    serverMode: str
    serverRole: str
    serverId: str


class AuthStats(TypedDict):
    storage: dict
    replication: dict
    cluster: dict


class AuthResponseData(TypedDict):
    token: str
    info: AuthInfo
    stats: AuthStats
    features: list[str]
    vocabulary: dict


class AuthResponse(TypedDict):
    data: AuthResponseData


class LoginRequest(BaseModel):
    host: str = "host.docker.internal"
    port: int = 7687
    username: str = ""
    password: str = ""
    isRouted: bool = False
    isEncrypted: bool = False
    wsPort: int = 7444
    databaseName: str = ""
    appId: str = "e795be3e-a0ec-48a5-bad3-e485640dceca"


class CypherQueryRequest(BaseModel):
    query: str
    tone_name: str = "an individual"
    num_of_queries: int = 5


# Query Graph

class GraphNodeProperties(TypedDict):
    __mg_id__: int


class GraphNode(TypedDict):
    id: int
    labels: List[str]
    properties: Dict[str, Any]
    type: str


class RelationshipProperties(TypedDict):
    pass  # This can be extended as per the actual properties of relationships.


class Relationship(TypedDict):
    id: int
    start: int
    end: int
    label: str
    properties: RelationshipProperties
    type: str


class Path(TypedDict):
    id: Optional[int]
    nodes: List[GraphNode]
    relationships: List[Relationship]
    type: str


class Record(TypedDict):
    path: Path


class Query(TypedDict):
    text: str
    parameters: Dict[str, str]


class Notification(TypedDict):
    code: str
    title: str
    description: str
    severity: str
    position: Dict


class Server(TypedDict):
    id: str
    type: str
    address: str
    protocolVersion: float


class Database(TypedDict):
    name: str


class Stats(TypedDict):
    nodesCreated: int
    nodesDeleted: int
    relationshipsCreated: int
    relationshipsDeleted: int
    propertiesSet: int
    labelsAdded: int
    labelsRemoved: int
    indexesAdded: int
    indexesRemoved: int
    constraintsAdded: int
    constraintsRemoved: int


class Summary(TypedDict):
    query: Query
    queryType: str
    notifications: List[Notification]
    server: Server
    database: Database
    costEstimate: int
    parsingTime: float
    planExecutionTime: float
    planningTime: float
    stats: Stats


class GraphResponseData(TypedDict):
    type: str
    records: List[Record]
    summary: Summary


class GraphQueryMetadata(BaseModel):
    queryId: str
    source: str = "lab-user"


class GraphQueryRequest(BaseModel):
    query: str
    metadata: GraphQueryMetadata


class GraphQueryResponse(TypedDict):
    data: GraphResponseData
