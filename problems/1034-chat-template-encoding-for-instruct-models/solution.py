def encode(text):
    if text is None:
        return []
    return [ord(c) for c in text]

_marker_begin_of_text = [1]
_marker_start_header_id = [2]
_marker_end_header_id = [3]
_marker_eot_id = [4]

def encode_header(role: str) -> list:
    return _marker_start_header_id + encode(role) + _marker_end_header_id + encode("\n\n")

def encode_chat(user_message: str, system_message: str = None) -> list:
    if system_message is None:
        system_message = "You are a helpful assistant."
    return (
        _marker_begin_of_text
        + encode_header("system") + encode(system_message) + _marker_eot_id
        + encode_header("user") + encode(user_message) + _marker_eot_id
        + encode_header("assistant")
    )