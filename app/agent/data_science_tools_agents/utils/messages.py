def get_tool_call_names(messages):
    """Method to extract the tool call names from a list of LangChain messages.

    Parameters
    ----------
    messages : list
        A list of LangChain messages.

    Returns
    -------
    tool_calls : list
        A list of tool call names.

    """
    tool_calls = []
    for message in messages:
        try:
            if "tool_call_id" in list(dict(message).keys()):
                tool_calls.append(message.name)
        except:
            pass
    return tool_calls
