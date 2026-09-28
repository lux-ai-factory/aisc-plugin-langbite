# Fixed conversation histories sent before a test prompt.
# A history is a list of {"role", "content"} messages in chat format.

ROLES = ('system', 'user', 'assistant')


def parse_histories(raw: list) -> dict[str, list[dict]]:
    histories = {}
    for item in raw:
        name = item.get('name')
        if not name or not str(name).strip():
            raise ValueError('History template without a name')
        if name in histories:
            raise ValueError(f'Duplicate history template name: {name}')
        histories[name] = validate_messages(name, item.get('messages'))
    return histories


def validate_messages(name, messages) -> list[dict]:
    if not messages:
        raise ValueError(f'History {name}: no messages')
    result = []
    for i, message in enumerate(messages):
        role = message.get('role')
        content = message.get('content')
        if role not in ROLES:
            raise ValueError(f'History {name}: message {i} has invalid role {role!r}')
        if not isinstance(content, str) or not content.strip():
            raise ValueError(f'History {name}: message {i} has empty content')
        if role == 'system' and i > 0:
            raise ValueError(f'History {name}: a system message may only come first')
        result.append({'role': role, 'content': content})
    if result[-1]['role'] == 'user':
        raise ValueError(f'History {name}: the last message cannot be a user message, the test prompt follows it')
    return result
