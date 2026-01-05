import re

class MarkdownSplitter:
    def __init__(self, markdown_content, split_paragraphs=False):
        self.markdown_content = markdown_content.splitlines()
        self.sections = []
        self.split_paragraphs = split_paragraphs  # New parameter to control paragraph splitting
    
    def is_heading(self, line):
        """Returns the heading level if the line is a heading, otherwise returns None."""
        match = re.match(r'^(#{1,4})\s', line)
        return len(match.group(1)) if match else None

    def split(self):
        current_hierarchy = []  # Stores the current heading hierarchy
        current_paragraph = []

        i = 0
        while i < len(self.markdown_content):
            line = self.markdown_content[i].strip()  # Remove leading/trailing whitespace
            
            if not line:  # Empty line found
                if self.split_paragraphs:  # Only handle splitting when split_paragraphs is True
                    # Check the next non-empty line
                    next_non_empty_line = None
                    for j in range(i + 1, len(self.markdown_content)):
                        if self.markdown_content[j].strip():  # Find the next non-empty line
                            next_non_empty_line = self.markdown_content[j].strip()
                            break
                    
                    # If the next non-empty line is a heading or not starting with '#', split paragraph
                    if next_non_empty_line and (self.is_heading(next_non_empty_line) or not next_non_empty_line.startswith('#')) and len(current_paragraph) > 0:
                        # Add the paragraph with the current hierarchy
                        self.sections.append("\n".join(current_hierarchy + ["\n".join(current_paragraph)]))
                        current_paragraph = []  # Reset for the next paragraph

                i += 1
                continue
            
            heading_level = self.is_heading(line)
            
            if heading_level:
                # If we encounter a heading, finalize the current paragraph
                if current_paragraph:
                    # Add the paragraph with the current hierarchy
                    self.sections.append("\n".join(current_hierarchy + ["\n".join(current_paragraph)]))
                    current_paragraph = []

                # Adjust the hierarchy based on the heading level
                # Keep only the parts of the hierarchy up to the current heading level
                current_hierarchy = current_hierarchy[:heading_level - 1] + [line]
            else:
                # Regular content: append the line to the current paragraph
                current_paragraph.append(line)

            i += 1

        # Finalize the last paragraph if present
        if current_paragraph:
            self.sections.append("\n".join(current_hierarchy + ["\n".join(current_paragraph)]))

        return self.sections

