INTERNAL_TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "multiply_by_two",
            "description": "Вызывай меня для умножения на 2",
            "parameters": {
                "type": "object",
                "properties": {
                    "number": {"type": "number"}
                },
                "required": ["number"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "divide",
            "description": "Вызывай меня для деления",
            "parameters": {
                "type": "object",
                "properties": {
                    "a": {"type": "number"},
                    "b": {"type": "number"}
                },
                "required": ["a", "b"]
            }
        }
    }
]


def call_internal_tool(name: str, args: dict):
    if name == "multiply_by_two":
        return args["number"] * 2
    elif name == "divide":
        if args["b"] == 0:
            return "Error: Деление на 0"
        return args["a"] / args["b"]
    return "Error: Что вы хотите?"
