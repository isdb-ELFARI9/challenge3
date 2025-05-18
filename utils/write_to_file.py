# file_writer.py

def write_to_file(filename, content, append=True):
    """
    Opens a file and writes the specified content to it.
    
    Args:
        filename (str): The name of the file to write to
        content (str): The content to write to the file
        append (bool, optional): If True, append to the file instead of overwriting. Defaults to False.
        
    Returns:
        bool: True if writing was successful, False otherwise
    """
    try:
        # Use 'a' mode if append is True, otherwise use 'w' mode
        mode = 'a' if append else 'w'
        with open(filename, mode) as file:
            file.write(content)
        return True
    except Exception as e:
        print(f"Error writing to file: {e}")
        return False


# This allows the function to be imported properly from other files
if __name__ == "__main__":
    # Example usage when running this file directly:
    write_to_file("example.txt", "Hello, this is some text I want to write to a file!")
    # Append more text
    write_to_file("example.txt", "\nThis is additional text appended to the file.", append=True)