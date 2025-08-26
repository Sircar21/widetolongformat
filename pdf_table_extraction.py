import fitz  # PyMuPDF
import pandas as pd
import re
from typing import List, Tuple, Optional, Dict, Any
import json
import io

class SectionBasedPDFExtractor:
    """
    Extract data from complex PDF tables using section-based approach
    """
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
    
    # ==================== APPROACH 1: TEXT-BASED EXTRACTION ====================
    
    def extract_by_text_search(self, page_no: int, section_identifier: str, 
                              column_name: str, row_name: str) -> Optional[str]:
        """
        Extract data by searching through raw text within a specific section
        Works well when tables are text-based but poorly structured
        """
        page = self.doc[page_no - 1]
        text = page.get_text()
        
        print(f"=== TEXT SEARCH METHOD ===")
        print(f"Looking for section: '{section_identifier}'")
        print(f"Target: Row='{row_name}', Column='{column_name}'")
        
        # Split text into lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        # Find the section
        section_start, section_end = self._find_section_boundaries(lines, section_identifier)
        if section_start == -1:
            print("Section identifier not found in text")
            return None
        
        print(f"Found section from line {section_start} to {section_end}")
        
        # Extract section content
        section_lines = lines[section_start:section_end]
        
        # Find table data within the section
        table_lines = self._extract_table_lines_from_section(section_lines)
        
        # Parse the table structure
        result = self._parse_text_table(table_lines, column_name, row_name)
        return result
    
    def _find_section_boundaries(self, lines: List[str], section_name: str) -> Tuple[int, int]:
        """Find start and end of a section"""
        section_name_lower = section_name.lower()
        section_start = -1
        section_end = len(lines)
        
        # Find section start
        for i, line in enumerate(lines):
            line_lower = line.lower()
            
            # Look for section headers (various formats)
            if (section_name_lower in line_lower and 
                (line.isupper() or  # ALL CAPS sections
                 line.startswith(section_name) or  # Exact start
                 re.match(r'^[\d\.]*\s*' + re.escape(section_name), line, re.IGNORECASE) or  # Numbered sections
                 len(line.split()) <= 5)):  # Short lines are likely headers
                section_start = i
                print(f"Section start found at line {i}: '{line}'")
                break
        
        if section_start == -1:
            # Fallback: just look for the section name anywhere
            for i, line in enumerate(lines):
                if section_name_lower in line.lower():
                    section_start = i
                    print(f"Section found (fallback) at line {i}: '{line}'")
                    break
        
        if section_start == -1:
            return -1, -1
        
        # Find section end (next section or significant break)
        for i in range(section_start + 1, len(lines)):
            line = lines[i]
            
            # Check for next section indicators
            if (len(line.split()) <= 5 and  # Short line (likely header)
                (line.isupper() or  # ALL CAPS
                 re.match(r'^[\d\.]+\s+[A-Z]', line) or  # Numbered section
                 any(keyword in line.lower() for keyword in 
                     ['section', 'chapter', 'part', 'appendix', 'summary', 'conclusion']))):
                
                # Make sure it's not just a table header within our section
                if i - section_start > 10:  # Only if we have enough content
                    section_end = i
                    print(f"Section end found at line {i}: '{line}'")
                    break
        
        return section_start, section_end
    
    def _extract_table_lines_from_section(self, section_lines: List[str]) -> List[str]:
        """Extract lines that likely contain table data from section"""
        table_lines = []
        
        # Skip the section header line(s)
        start_idx = 1 if section_lines else 0
        
        for i, line in enumerate(section_lines[start_idx:], start_idx):
            # Include lines that look like table data
            if self._looks_like_table_row(line):
                table_lines.append(line)
            
            # Also include lines that might be headers or row names
            elif (len(line.split()) >= 2 and 
                  not line.endswith(':') and  # Skip intro text
                  not line.lower().startswith('the ')):  # Skip descriptive text
                table_lines.append(line)
        
        return table_lines
    
    def _looks_like_table_row(self, line: str) -> bool:
        """Check if a line looks like it contains table data"""
        # Has multiple words/numbers separated by spaces
        parts = line.split()
        if len(parts) < 2:
            return False
        
        # Contains numbers or common table elements
        has_numbers = any(part.replace(',', '').replace('.', '').replace('%', '').replace('$', '').isdigit() 
                         for part in parts)
        
        # Or contains typical table formatting
        has_table_chars = any(char in line for char in ['|', '\t', '  '])
        
        return has_numbers or has_table_chars
    
    def _parse_text_table(self, lines: List[str], column_name: str, row_name: str) -> Optional[str]:
        """Parse text-based table to find specific data point"""
        if not lines:
            return None
        
        # Find header row
        header_row = None
        header_idx = -1
        
        for i, line in enumerate(lines):
            if column_name.lower() in line.lower():
                header_row = line
                header_idx = i
                break
        
        if header_row is None:
            print(f"Column '{column_name}' not found in headers")
            return None
        
        # Parse header to find column position
        column_position = self._find_column_position(header_row, column_name)
        if column_position == -1:
            return None
        
        # Find data row
        for line in lines[header_idx + 1:]:
            if row_name.lower() in line.lower():
                # Extract value at column position
                value = self._extract_value_at_position(line, column_position)
                if value:
                    return value
        
        print(f"Row '{row_name}' not found")
        return None
    
    def _find_column_position(self, header_row: str, column_name: str) -> int:
        """Find the position of column in header"""
        # Simple approach: find where the column name appears
        start_pos = header_row.lower().find(column_name.lower())
        if start_pos != -1:
            return start_pos
        return -1
    
    def _extract_value_at_position(self, line: str, position: int) -> Optional[str]:
        """Extract value from line at approximate position"""
        # Split by multiple spaces to get columns
        parts = re.split(r'\s{2,}', line.strip())
        
        # If position-based extraction fails, try pattern matching
        if len(parts) >= 2:
            # Look for numbers in the parts
            for part in parts[1:]:  # Skip first part (usually row name)
                if re.search(r'[\d,.$%]+', part):
                    return part.strip()
        
        return None
    
    # ==================== APPROACH 2: COORDINATE-BASED EXTRACTION ====================
    
    def extract_by_coordinates(self, page_no: int, section_identifier: str,
                              column_name: str, row_name: str) -> Optional[str]:
        """
        Extract data using coordinate-based approach within a section
        Find text positions and extract based on spatial relationships
        """
        page = self.doc[page_no - 1]
        
        print(f"=== COORDINATE-BASED METHOD ===")
        
        # Get all text blocks with coordinates
        text_dict = page.get_text("dict")
        
        # Find section region first
        section_region = self._find_section_region(text_dict, section_identifier)
        if section_region is None:
            print("Section identifier not found")
            return None
        
        print(f"Section region found: {section_region}")
        
        # Find column header location within section
        column_rect = self._find_text_location(text_dict, column_name, search_area=section_region)
        if column_rect is None:
            print(f"Column '{column_name}' not found in section")
            return None
        
        # Find row name location within section
        row_rect = self._find_text_location(text_dict, row_name, search_area=section_region)
        if row_rect is None:
            print(f"Row '{row_name}' not found in section")
            return None
        
        # Find intersection point (where row and column meet)
        value = self._find_value_at_intersection(text_dict, row_rect, column_rect)
        return value
    
    def _find_section_region(self, text_dict: dict, section_name: str) -> Optional[fitz.Rect]:
        """Find the bounding region of a section"""
        section_name_lower = section_name.lower()
        
        # Find the section header
        section_block = None
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    for span in line["spans"]:
                        line_text += span["text"]
                    
                    if (section_name_lower in line_text.lower() and
                        (len(line_text.split()) <= 6 or  # Short lines are likely headers
                         line_text.isupper() or  # ALL CAPS
                         line_text.strip().startswith(section_name))):  # Starts with section name
                        
                        bbox = line["bbox"]
                        section_block = fitz.Rect(bbox)
                        print(f"Section header found: '{line_text.strip()}'")
                        break
            
            if section_block:
                break
        
        if not section_block:
            # Fallback: just find any mention of section name
            for block in text_dict["blocks"]:
                if "lines" in block:
                    for line in block["lines"]:
                        for span in line["spans"]:
                            if section_name_lower in span["text"].lower():
                                bbox = span["bbox"]
                                section_block = fitz.Rect(bbox)
                                break
        
        if not section_block:
            return None
        
        # Estimate section region (area below the section header)
        # Assume section extends to right edge of page and down significantly
        page_rect = fitz.Rect(0, 0, 612, 792)  # Standard page size, adjust if needed
        
        section_region = fitz.Rect(
            max(0, section_block.x0 - 50),  # Start a bit to the left
            section_block.y0,  # Start at section header
            page_rect.x1,  # Extend to page width
            min(page_rect.y1, section_block.y1 + 400)  # Extend down 400 points
        )
        
        return section_region
    
    def _find_text_location(self, text_dict: dict, search_text: str, 
                           search_area: fitz.Rect = None) -> Optional[fitz.Rect]:
        """Find location of text in PDF"""
        search_text_lower = search_text.lower()
        
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].lower()
                        if search_text_lower in text:
                            bbox = span["bbox"]
                            rect = fitz.Rect(bbox)
                            
                            # Check if within search area
                            if search_area is None or rect.intersects(search_area):
                                return rect
        
        return None
    
    def _find_value_at_intersection(self, text_dict: dict, row_rect: fitz.Rect, 
                                   column_rect: fitz.Rect) -> Optional[str]:
        """Find value at intersection of row and column"""
        # Look for text near the intersection point
        target_x = column_rect.x0 + (column_rect.x1 - column_rect.x0) / 2
        target_y = row_rect.y0 + (row_rect.y1 - row_rect.y0) / 2
        
        closest_text = None
        min_distance = float('inf')
        
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        bbox = span["bbox"]
                        span_x = bbox[0] + (bbox[2] - bbox[0]) / 2
                        span_y = bbox[1] + (bbox[3] - bbox[1]) / 2
                        
                        # Calculate distance from target point
                        distance = ((span_x - target_x) ** 2 + (span_y - target_y) ** 2) ** 0.5
                        
                        # Must be reasonably close and contain data-like content
                        if distance < min_distance and distance < 50:  # 50 points threshold
                            text = span["text"].strip()
                            if text and (text.replace(',', '').replace('.', '').replace('%', '').replace('$', '').isdigit() or 
                                       re.search(r'[\d,.$%]+', text)):
                                closest_text = text
                                min_distance = distance
        
        return closest_text
    
    # ==================== APPROACH 3: REGION-BASED EXTRACTION ====================
    
    def extract_by_region_analysis(self, page_no: int, section_identifier: str,
                                  column_name: str, row_name: str) -> Optional[str]:
        """
        Analyze page regions and extract text blocks spatially within a section
        Good for tables with clear spatial structure
        """
        print(f"=== REGION-BASED METHOD ===")
        
        page = self.doc[page_no - 1]
        
        # Get text blocks with their positions
        blocks = page.get_text("dict")["blocks"]
        text_blocks = []
        
        for block in blocks:
            if "lines" in block:
                for line in block["lines"]:
                    for span in line["spans"]:
                        text_blocks.append({
                            'text': span["text"].strip(),
                            'bbox': span["bbox"],
                            'x0': span["bbox"][0],
                            'y0': span["bbox"][1], 
                            'x1': span["bbox"][2],
                            'y1': span["bbox"][3]
                        })
        
        # Find section region
        section_region = self._find_section_region_blocks(text_blocks, section_identifier)
        if not section_region:
            print("Section region not found")
            return None
        
        print(f"Section region: {section_region}")
        
        # Filter blocks within section region
        section_blocks = self._get_blocks_in_region(text_blocks, section_region)
        print(f"Found {len(section_blocks)} blocks in section")
        
        # Find column and row positions within section
        column_x = self._find_column_x_position(section_blocks, column_name)
        row_y = self._find_row_y_position(section_blocks, row_name)
        
        if column_x is None or row_y is None:
            print(f"Could not locate column_x={column_x} or row_y={row_y}")
            return None
        
        # Find value at intersection
        value = self._find_value_at_position(section_blocks, column_x, row_y)
        return value
    
    def _find_section_region_blocks(self, text_blocks: List[dict], section_name: str) -> Optional[dict]:
        """Find the bounding region of the section"""
        section_name_lower = section_name.lower()
        
        # Find the section header block
        section_block = None
        for block in text_blocks:
            block_text = block['text'].lower()
            
            # Look for section header characteristics
            if (section_name_lower in block_text and
                (len(block['text'].split()) <= 6 or  # Short text
                 block['text'].isupper() or  # ALL CAPS
                 block['text'].strip().startswith(section_name))):  # Starts with section name
                section_block = block
                print(f"Section header block found: '{block['text']}'")
                break
        
        if not section_block:
            # Fallback: find any block containing section name
            for block in text_blocks:
                if section_name_lower in block['text'].lower():
                    section_block = block
                    break
        
        if not section_block:
            return None
        
        # Estimate section boundaries
        section_region = {
            'x0': max(0, section_block['x0'] - 50),  # Start a bit to the left
            'y0': section_block['y0'],  # Start at section header
            'x1': section_block['x1'] + 300,  # Extend to the right
            'y1': section_block['y1'] + 400   # Extend down
        }
        
        return section_region
    
    def _get_blocks_in_region(self, text_blocks: List[dict], region: dict) -> List[dict]:
        """Get all text blocks within a region"""
        filtered_blocks = []
        
        for block in text_blocks:
            # Check if block overlaps with region
            if (block['x1'] > region['x0'] and block['x0'] < region['x1'] and
                block['y1'] > region['y0'] and block['y0'] < region['y1']):
                filtered_blocks.append(block)
        
        return filtered_blocks
    
    def _find_column_x_position(self, blocks: List[dict], column_name: str) -> Optional[float]:
        """Find x-coordinate of column"""
        column_name_lower = column_name.lower()
        
        for block in blocks:
            if column_name_lower in block['text'].lower():
                return block['x0']  # Return left edge of column header
        
        return None
    
    def _find_row_y_position(self, blocks: List[dict], row_name: str) -> Optional[float]:
        """Find y-coordinate of row"""
        row_name_lower = row_name.lower()
        
        for block in blocks:
            if row_name_lower in block['text'].lower():
                return block['y0']  # Return top edge of row
        
        return None
    
    def _find_value_at_position(self, blocks: List[dict], target_x: float, target_y: float) -> Optional[str]:
        """Find value near the target position"""
        best_block = None
        min_distance = float('inf')
        
        for block in blocks:
            # Calculate distance from target position
            block_x = block['x0']
            block_y = block['y0']
            
            distance = ((block_x - target_x) ** 2 + (block_y - target_y) ** 2) ** 0.5
            
            # Must contain numeric data and be reasonably close
            if (distance < min_distance and distance < 100 and
                block['text'] and re.search(r'[\d,.$%]+', block['text'])):
                best_block = block
                min_distance = distance
        
        return best_block['text'] if best_block else None
    
    # ==================== APPROACH 4: LINE-BY-LINE PARSING ====================
    
    def extract_by_line_parsing(self, page_no: int, section_identifier: str,
                               column_name: str, row_name: str) -> Optional[str]:
        """
        Parse PDF line by line to reconstruct table structure within a section
        Good for tables that appear structured in text but not in table format
        """
        print(f"=== LINE PARSING METHOD ===")
        
        page = self.doc[page_no - 1]
        
        # Get text with layout information
        text_dict = page.get_text("dict")
        
        # Extract lines with their y-coordinates
        lines = self._extract_lines_with_coords(text_dict)
        
        # Find section boundaries
        section_lines = self._find_section_lines(lines, section_identifier)
        if not section_lines:
            print("Section lines not found")
            return None
        
        print(f"Found {len(section_lines)} lines in section")
        
        # Parse table structure from section lines
        table_data = self._parse_table_from_section_lines(section_lines)
        
        # Find the specific data point
        value = self._find_data_in_parsed_table(table_data, column_name, row_name)
        return value
    
    def _extract_lines_with_coords(self, text_dict: dict) -> List[dict]:
        """Extract text lines with their coordinates"""
        lines = []
        
        for block in text_dict["blocks"]:
            if "lines" in block:
                for line in block["lines"]:
                    line_text = ""
                    spans_data = []
                    
                    for span in line["spans"]:
                        line_text += span["text"]
                        spans_data.append({
                            'text': span["text"],
                            'x': span["bbox"][0],
                            'y': span["bbox"][1]
                        })
                    
                    if line_text.strip():
                        lines.append({
                            'text': line_text.strip(),
                            'y': line["bbox"][1],
                            'spans': spans_data
                        })
        
        # Sort by y-coordinate (top to bottom)
        lines.sort(key=lambda x: x['y'])
        return lines
    
    def _find_section_lines(self, lines: List[dict], section_name: str) -> List[dict]:
        """Find lines that belong to the section"""
        section_name_lower = section_name.lower()
        
        # Find section start
        start_idx = -1
        for i, line in enumerate(lines):
            line_text_lower = line['text'].lower()
            
            # Look for section header
            if (section_name_lower in line_text_lower and
                (len(line['text'].split()) <= 6 or  # Short lines are likely headers
                 line['text'].isupper() or  # ALL CAPS
                 line['text'].strip().startswith(section_name))):  # Starts with section name
                start_idx = i
                print(f"Section start at line {i}: '{line['text']}'")
                break
        
        if start_idx == -1:
            # Fallback: find any line with section name
            for i, line in enumerate(lines):
                if section_name_lower in line['text'].lower():
                    start_idx = i
                    print(f"Section found (fallback) at line {i}: '{line['text']}'")
                    break
        
        if start_idx == -1:
            return []
        
        # Find section end
        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            line = lines[i]
            line_text = line['text']
            
            # Stop at next section
            if (len(line_text.split()) <= 5 and  # Short line
                (line_text.isupper() or  # ALL CAPS
                 re.match(r'^[\d\.]+\s+[A-Z]', line_text) or  # Numbered section
                 any(keyword in line_text.lower() for keyword in 
                     ['section', 'chapter', 'part', 'appendix']))):
                
                if i - start_idx > 5:  # Only if we have reasonable content
                    end_idx = i
                    print(f"Section end at line {i}: '{line_text}'")
                    break
        
        return lines[start_idx:end_idx]
    
    def _parse_table_from_section_lines(self, lines: List[dict]) -> List[List[str]]:
        """Parse table structure from section lines"""
        table_data = []
        
        # Skip the section header
        start_idx = 1 if lines else 0
        
        for line in lines[start_idx:]:
            # Skip obvious non-table lines
            if (line['text'].endswith(':') or  # Descriptive text
                line['text'].lower().startswith('the ') or  # Prose
                len(line['text'].split()) > 15):  # Long sentences
                continue
            
            # Try to split line into columns
            row_data = self._split_line_into_columns(line)
            if row_data and len(row_data) > 1:
                table_data.append(row_data)
        
        return table_data
    
    def _split_line_into_columns(self, line: dict) -> List[str]:
        """Split a line into columns based on spacing"""
        text = line['text']
        
        # Method 1: Split by multiple spaces
        parts = re.split(r'\s{2,}', text.strip())
        if len(parts) > 1:
            return [part.strip() for part in parts if part.strip()]
        
        # Method 2: Split by tabs
        if '\t' in text:
            return [part.strip() for part in text.split('\t') if part.strip()]
        
        # Method 3: Use span positions if available
        if 'spans' in line and len(line['spans']) > 1:
            return [span['text'].strip() for span in line['spans'] if span['text'].strip()]
        
        return [text.strip()] if text.strip() else []
    
    def _find_data_in_parsed_table(self, table_data: List[List[str]], 
                                  column_name: str, row_name: str) -> Optional[str]:
        """Find specific data in parsed table"""
        if not table_data:
            return None
        
        # Find header row and column index
        header_row = None
        column_idx = None
        
        for row in table_data[:3]:  # Check first 3 rows for headers
            for i, cell in enumerate(row):
                if column_name.lower() in cell.lower():
                    header_row = row
                    column_idx = i
                    break
            if column_idx is not None:
                break
        
        if column_idx is None:
            print(f"Column '{column_name}' not found in headers")
            return None
        
        # Find data row
        for row in table_data:
            if row and row_name.lower() in row[0].lower():
                if len(row) > column_idx:
                    return row[column_idx]
        
        print(f"Row '{row_name}' not found in data")
        return None

    # ==================== APPROACH 5: PATTERN MATCHING ====================
    
    def extract_by_pattern_matching(self, page_no: int, section_identifier: str,
                                   column_name: str, row_name: str) -> Optional[str]:
        """
        Use regex patterns to find data in structured text within a section
        Good for consistently formatted documents
        """
        page = self.doc[page_no - 1]
        text = page.get_text()
        
        print(f"=== PATTERN MATCHING METHOD ===")
        
        # Extract section text first
        section_text = self._extract_section_text(text, section_identifier)
        if not section_text:
            print("Section not found in text")
            return None
        
        print(f"Section text length: {len(section_text)} characters")
        
        # Create patterns to find the data within the section
        patterns = [
            # Pattern 1: Row name followed by values in section
            rf'{re.escape(row_name)}\s*[:\-]?\s*([\d,.$%\s]+)',
            
            # Pattern 2: Table structure with column header and row
            rf'{re.escape(column_name)}.*?\n.*?{re.escape(row_name)}\s*[:\-]?\s*([\d,.$%]+)',
            
            # Pattern 3: Multi-column table pattern
            rf'{re.escape(row_name)}\s*[:\-]?.*?{re.escape(column_name)}.*?([\d,.$%]+)',
            
            # Pattern 4: Simple value after row name
            rf'{re.escape(row_name)}\s*[:\-]?\s*([^\n]*(?:[\d,.$%]+)[^\n]*)',
        ]
        
        for i, pattern in enumerate(patterns):
            print(f"Trying pattern {i+1}: {pattern[:50]}...")
            matches = re.search(pattern, section_text, re.IGNORECASE | re.DOTALL)
            if matches:
                value = matches.group(1).strip()
                print(f"Pattern {i+1} matched: '{value}'")
                
                # Clean up the extracted value
                cleaned_value = self._extract_number_from_text(value)
                if cleaned_value:
                    return cleaned_value
        
        return None
    
    def _extract_section_text(self, full_text: str, section_name: str) -> Optional[str]:
        """Extract text content of a specific section"""
        section_name_lower = section_name.lower()
        lines = full_text.split('\n')
        
        # Find section start
        start_idx = -1
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            if (section_name_lower in line_lower and
                (line.isupper() or  # ALL CAPS section
                 len(line.split()) <= 6 or  # Short header
                 re.match(r'^[\d\.]*\s*' + re.escape(section_name), line, re.IGNORECASE))):  # Numbered section
                start_idx = i
                break
        
        if start_idx == -1:
            # Fallback: just find section name
            for i, line in enumerate(lines):
                if section_name_lower in line.lower():
                    start_idx = i
                    break
        
        if start_idx == -1:
            return None
        
        # Find section end
        end_idx = len(lines)
        for i in range(start_idx + 1, len(lines)):
            line = lines[i].strip()
            
            # Stop at next section
            if (len(line.split()) <= 5 and
                (line.isupper() or
                 re.match(r'^[\d\.]+\s+[A-Z]', line) or
                 any(keyword in line.lower() for keyword in 
                     ['section', 'chapter', 'part', 'appendix']))):
                
                if i - start_idx > 3:  # Only if reasonable content
                    end_idx = i
                    break
        
        section_text = '\n'.join(lines[start_idx:end_idx])
        return section_text
    
    def _extract_number_from_text(self, text: str) -> Optional[str]:
        """Extract the most likely number from text"""
        # Look for number patterns
        number_patterns = [
            r'[\d,]+\.?\d*%?',  # Regular numbers with commas and percentages
            r'\$[\d,]+\.?\d*',   # Dollar amounts
            r'[\d,]+\.?\d*',     # Simple numbers
        ]
        
        for pattern in number_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Return the longest match (most likely to be the main value)
                return max(matches, key=len)
        
        return None
    
    # ==================== MAIN EXTRACTION METHOD ====================
    
    def extract_data(self, page_no: int, section_identifier: str,
                    column_name: str, row_name: str,
                    methods: List[str] = None) -> Optional[str]:
        """
        Try multiple extraction methods in sequence for section-based extraction
        
        Available methods:
        - 'text_search': Search through raw text within section
        - 'coordinates': Use coordinate-based extraction within section
        - 'region_analysis': Analyze spatial regions within section
        - 'line_parsing': Parse section line by line
        - 'pattern': Use regex pattern matching within section
        """
        
        if methods is None:
            methods = ['text_search', 'line_parsing', 'region_analysis', 'pattern', 'coordinates']
        
        method_map = {
            'text_search': self.extract_by_text_search,
            'coordinates': self.extract_by_coordinates,
            'region_analysis': self.extract_by_region_analysis,
            'line_parsing': self.extract_by_line_parsing,
            'pattern': self.extract_by_pattern_matching
        }
        
        for method_name in methods:
            if method_name not in method_map:
                continue
                
            try:
                print(f"\n--- Trying {method_name.upper()} method ---")
                result = method_map[method_name](page_no, section_identifier, column_name, row_name)
                
                if result:
                    print(f"✓ SUCCESS with {method_name}: {result}")
                    return result
                else:
                    print(f"✗ {method_name} returned no result")
                    
            except Exception as e:
                print(f"✗ {method_name} failed with error: {e}")
                continue
        
        print("All methods failed")
        return None
    
    def debug_page_content(self, page_no: int, section_name: str = None):
        """Debug method to see page content structure"""
        page = self.doc[page_no - 1]
        
        print(f"=== DEBUG PAGE {page_no} CONTENT ===")
        
        # Method 1: Show raw text
        print("\n--- RAW TEXT ---")
        text = page.get_text()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for i, line in enumerate(lines[:30]):  # First 30 lines
            print(f"{i:2d}: {line}")
        
        # Method 2: Show text blocks with coordinates
        print("\n--- TEXT BLOCKS WITH COORDINATES ---")
        text_dict = page.get_text("dict")
        block_count = 0
        
        for block in text_dict["blocks"]:
            if "lines" in block and block_count < 15:  # First 15 blocks
                print(f"\nBlock {block_count}:")
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            bbox = span["bbox"]
                            print(f"  [{bbox[0]:.1f}, {bbox[1]:.1f}] '{span['text'].strip()}'")
                block_count += 1
        
        # Method 3: Show section-specific info if section name provided
        if section_name:
            print(f"\n--- SECTION ANALYSIS FOR '{section_name}' ---")
            section_start, section_end = self._find_section_boundaries(lines, section_name)
            if section_start != -1:
                print(f"Section found from line {section_start} to {section_end}")
                print("Section content:")
                for i in range(section_start, min(section_end, section_start + 20)):
                    print(f"  {i:2d}: {lines[i]}")
            else:
                print("Section not found")
        
        # Method 4: Try to find potential tables
        print("\n--- POTENTIAL TABLE INDICATORS ---")
        table_indicators = []
        for line in lines:
            # Look for lines that might be table headers or contain multiple values
            if (len(line.split()) > 3 and 
                (any(char in line for char in ['|', '\t']) or
                 len([part for part in re.split(r'\s+', line) if re.search(r'\d', part)]) > 2)):
                table_indicators.append(line)
        
        for indicator in table_indicators[:10]:
            print(f"  {indicator}")
        
        if not table_indicators:
            print("  No obvious table structure found")
    
    def close(self):
        """Close the PDF document"""
        self.doc.close()

# ==================== USAGE EXAMPLES ====================

def extract_with_debug(pdf_path: str, config: dict):
    """Extract data with debugging information"""
    extractor = SectionBasedPDFExtractor(pdf_path)
    
    try:
        # Show debug info first
        if config.get('debug', False):
            extractor.debug_page_content(
                config['page_no'], 
                config.get('section_identifier')
            )
        
        # Try extraction
        result = extractor.extract_data(
            page_no=config['page_no'],
            section_identifier=config['section_identifier'],
            column_name=config['column_name'],
            row_name=config['row_name'],
            methods=config.get('methods', None)
        )
        
        return result
        
    finally:
        extractor.close()

def batch_extract(pdf_path: str, extraction_configs: List[dict]):
    """Extract multiple data points from a PDF"""
    extractor = SectionBasedPDFExtractor(pdf_path)
    results = {}
    
    try:
        for config in extraction_configs:
            data_point_name = config.get('name', f"data_point_{len(results)}")
            print(f"\n{'='*50}")
            print(f"EXTRACTING: {data_point_name}")
            print(f"{'='*50}")
            
            extracted_value = extractor.extract_data(
                page_no=config['page_no'],
                section_identifier=config['section_identifier'],
                column_name=config['column_name'],
                row_name=config['row_name'],
                methods=config.get('methods', None)
            )
            
            results[data_point_name] = extracted_value
            print(f"Result for {data_point_name}: {extracted_value}")
            
    finally:
        extractor.close()
    
    return results

# Example usage
if __name__ == "__main__":
    # Single extraction with debugging
    config = {
        'page_no': 1,
        'section_identifier': 'Financial Performance',  # Section name instead of table name
        'column_name': 'Q1 2024', 
        'row_name': 'Total Revenue',
        'methods': ['text_search', 'line_parsing', 'pattern'],  # Specify methods to try
        'debug': True  # Set to True to see page structure
    }
    
    # result = extract_with_debug('sample.pdf', config)
    # print(f"Final result: {result}")
    
    # Multiple extractions example
    extraction_configs = [
        {
            'name': 'total_revenue_q1',
            'page_no': 1,
            'section_identifier': 'Financial Performance',
            'column_name': 'Q1 2024',
            'row_name': 'Total Revenue'
        },
        {
            'name': 'net_income_q2',
            'page_no': 1,
            'section_identifier': 'Financial Performance', 
            'column_name': 'Q2 2024',
            'row_name': 'Net Income'
        },
        {
            'name': 'expenses_q1',
            'page_no': 2,
            'section_identifier': 'Operating Expenses',
            'column_name': 'Q1 2024',
            'row_name': 'Total Expenses'
        }
    ]
    
    # results = batch_extract('sample.pdf', extraction_configs)
    # print("\nAll Results:")
    # for name, value in results.items():
    #     print(f"{name}: {value}")
    
    # Just debug page structure
    # extractor = SectionBasedPDFExtractor('sample.pdf')
    # extractor.debug_page_content(1, 'Financial Performance')
    # extractor.close()
