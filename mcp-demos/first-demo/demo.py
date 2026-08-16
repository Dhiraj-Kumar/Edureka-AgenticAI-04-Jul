from mcp.server import MCPServer

mcp = MCPServer("Demo Server")

@mcp.tool()
def add(a: int, b: int) -> int:
    return a+b

if __name__ == "__main__":
    mcp.run()