This is a test.

# Filesystem Tool Instructions

The `filesystem` tool allows you to interact with the file system. Here are the available commands and how to use them:

*   **`filesystem.read_file(path)`**: Reads the contents of a file.
    *   `path` (string): The absolute path to the file.
*   **`filesystem.write_file(path, content)`**: Writes content to a file.
    *   `path` (string): The absolute path to the file.
    *   `content` (string): The content to write to the file.
*   **`filesystem.list_directory(path)`**: Lists the contents of a directory.
    *   `path` (string): The absolute path to the directory.

**Important:** All paths must be absolute and within the `/workspace` directory.