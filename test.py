def get_path():
    import traceback
    stack = traceback.extract_stack()
    (filename, line, procname, text) = stack[-1]
    return stack[-1]