from flask import url_for


def use_static_file(file_name: str) -> str:
    return url_for('static', filename=file_name)
