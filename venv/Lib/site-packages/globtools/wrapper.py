import glob
# decorators
def globfiles(func):
    def wrapper(*args, **kwargs):
        patterns = func(*args, **kwargs)
        files = []
        for pattern in patterns:
            files += glob.glob(pattern)
        return files
    return wrapper
