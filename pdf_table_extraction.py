import fitz  # PyMuPDF
import pandas as pd
import re
from typing import List, Tuple, Optional, Dict, Any
import json
import io

class AlternativePDFExtractor:
    """
    Alternative approaches for extracting data from complex PDF tables
    """
    
    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path
        self.doc = fitz.open(pdf_path)
    
    # ==================== APPROACH 1: TEXT-BASED EXTRACTION ====================
    
    def extract_by_text_search(self, page_no: int, table_identifier: str, 
                              column_name: str, row_name: str) -> Optional[str]:
        """
        Extract data by searching through raw text and using patterns
        Works well when tables are text-based but poorly structured
        """
        page = self.doc[page_no - 1]
        text = page.get_text()
        
        print(f"=== TEXT SEARCH METHOD ===")
        print(f"Looking for table: '{table_identifier}'")
        print(f"Target: Row='{row_name}', Column='{column_name}'")
        
        # Split text into lines
        lines = text.split('\n')
        lines = [line.strip() for line in lines if line.strip()]
        
        # Find the table section
        table_start = self._find_table_start_in_text(lines, table_identifier)
        if table_start == -1:
            print("Table identifier not found in text")
            return None
        
        # Extract table data from text
        table_lines = self._extract_table_lines(lines[table_start:])
        
        # Parse the table structure
        result = self._parse_text_table(table_lines, column_name, row_name)
        return result
    
    def _find_table_start_in_text(self, lines: List[str], identifier: str) -> int:
        """Find where the table starts in text"""
        identifier_lower = identifier.lower()
        
        for i, line in enumerate(lines):
            if identifier_lower in line.lower():
                return i
        
        return -1
    
    def _extract_table_lines(self, lines: List[str]) -> List[str]:
        """Extract lines that likely belong to the table"""
        table_lines = []
        
        for line in lines:
            # Stop at next section/table
            if any(keyword in line.lower() for keyword in ['table', 'figure', 'section', 'chapter']):
                if len(table_lines) > 5:  # Only stop if we have enough lines
                    break
            
            # Include lines that look like table data
            if self._looks_like_table_row(line):
                table_lines.append(line)
                
            # Stop if we hit too many empty lines
            if not line.strip() and len(table_lines) > 3:
                consecutive_empty = 1
                for next_line in lines[lines.index(line)+1:lines.index(line)+4]:
                    if not next_line.strip():
                        consecutive_empty += 1
                if consecutive_empty >= 3:
                    break
        
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
    
    def extract_by_coordinates(self, page_no: int, table_identifier: str,
                              column_name: str, row_name: str) -> Optional[str]:
        """
        Extract data using coordinate-based approach
        Find text positions and extract based on spatial relationships
        """
        page = self.doc[page_no - 1]
        
        print(f"=== COORDINATE-BASED METHOD ===")
        
        # Get all text blocks with coordinates
        text_dict = page.get_text("dict")
        
        # Find table identifier location
        table_rect = self._find_text_location(text_dict, table_identifier)
        if table_rect is None:
            print("Table identifier not found")
            return None
        
        # Find column header location
        column_rect = self._find_text_location(text_dict, column_name, search_area=table_rect)
        if column_rect is None:
            print(f"Column '{column_name}' not found")
            return None
        
        # Find row name location
        row_rect = self._find_text_location(text_dict, row_name, search_area=table_rect)
        if row_rect is None:
            print(f"Row '{row_name}' not found")
            return None
        
        # Find intersection point (where row and column meet)
        value = self._find_value_at_intersection(text_dict, row_rect, column_rect)
        return value
    
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
    
    def extract_by_region_analysis(self, page_no: int, table_identifier: str,
                                  column_name: str, row_name: str) -> Optional[str]:
        """
        Analyze page regions and extract text blocks spatially
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
        
        # Find table region
        table_region = self._find_table_region_blocks(text_blocks, table_identifier)
        if not table_region:
            print("Table region not found")
            return None
        
        # Filter blocks within table region
        table_blocks = self._get_blocks_in_region(text_blocks, table_region)
        
        # Find column and row positions
        column_x = self._find_column_x_position(table_blocks, column_name)
        row_y = self._find_row_y_position(table_blocks, row_name)
        
        if column_x is None or row_y is None:
            print(f"Could not locate column_x={column_x} or row_y={row_y}")
            return None
        
        # Find value at intersection
        value = self._find_value_at_position(table_blocks, column_x, row_y)
        return value
    
    def _find_table_region_blocks(self, text_blocks: List[dict], identifier: str) -> Optional[dict]:
        """Find the bounding region of the table"""
        identifier_lower = identifier.lower()
        
        # Find the identifier block
        identifier_block = None
        for block in text_blocks:
            if identifier_lower in block['text'].lower():
                identifier_block = block
                break
        
        if not identifier_block:
            return None
        
        # Estimate table region (area below the identifier)
        min_x = identifier_block['x0'] - 50
        max_x = identifier_block['x1'] + 200
        min_y = identifier_block['y0']
        max_y = identifier_block['y1'] + 300  # Assume table extends 300 points down
        
        return {'x0': min_x, 'y0': min_y, 'x1': max_x, 'y1': max_y}
    
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
    
    def extract_by_line_parsing(self, page_no: int, table_identifier: str,
                               column_name: str, row_name: str) -> Optional[str]:
        """
        Parse PDF line by line to reconstruct table structure
        Good for tables that appear structured in text but not in table format
        """
        print(f"=== LINE PARSING METHOD ===")
        
        page = self.doc[page_no - 1]
        
        # Get text with layout information
        text_dict = page.get_text("dict")
        
        # Extract lines with their y-coordinates
        lines = self._extract_lines_with_coords(text_dict)
        
        # Find table section
        table_lines = self._find_table_section(lines, table_identifier)
        if not table_lines:
            print("Table section not found")
            return None
        
        # Parse table structure
        table_data = self._parse_table_from_lines(table_lines)
        
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
    
    def _find_table_section(self, lines: List[dict], identifier: str) -> List[dict]:
        """Find lines that belong to the table"""
        identifier_lower = identifier.lower()
        
        # Find starting line
        start_idx = -1
        for i, line in enumerate(lines):
            if identifier_lower in line['text'].lower():
                start_idx = i
                break
        
        if start_idx == -1:
            return []
        
        # Extract table lines (until next section or empty space)
        table_lines = []
        for i in range(start_idx, len(lines)):
            line = lines[i]
            
            # Stop conditions
            if i > start_idx + 50:  # Maximum 50 lines
                break
                
            # Stop at next major section
            if (i > start_idx + 5 and 
                any(keyword in line['text'].lower() for keyword in 
                    ['section', 'table', 'figure', 'chapter', 'part'])):
                break
            
            table_lines.append(line)
        
        return table_lines
    
    def _parse_table_from_lines(self, lines: List[dict]) -> List[List[str]]:
        """Parse table structure from lines"""
        table_data = []
        
        for line in lines:
            # Skip obvious header/title lines
            if any(word in line['text'].lower() for word in ['table', 'figure']):
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
    
    def extract_by_pattern_matching(self, page_no: int, table_identifier: str,
                                   column_name: str, row_name: str) -> Optional[str]:
        """
        Use regex patterns to find data in structured text
        Good for consistently formatted documents
        """
        page = self.doc[page_no - 1]
        text = page.get_text()
        
        print(f"=== PATTERN MATCHING METHOD ===")
        
        # Create patterns based on the identifiers
        patterns = [
            # Pattern 1: Row name followed by values
            rf'{re.escape(row_name)}\s*[:\-]?\s*([\d,.$%\s]+)',
            
            # Pattern 2: Table with clear structure
            rf'{re.escape(table_identifier)}.*?{re.escape(row_name)}\s*[:\-]?\s*([\d,.$%]+)',
            
            # Pattern 3: Multi-line table pattern
            rf'{re.escape(column_name)}.*?\n.*?{re.escape(row_name)}\s*[:\-]?\s*([\d,.$%]+)',
        ]
        
        for pattern in patterns:
            matches = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if matches:
                value = matches.group(1).strip()
                # Clean up the extracted value
                value = re.search(r'[\d,.$%]+', value)
                if value:
                    return value.group(0)
        
        return None
    
    # ==================== MAIN EXTRACTION METHOD ====================
    
    def extract_data(self, page_no: int, table_identifier: str,
                    column_name: str, row_name: str,
                    methods: List[str] = None) -> Optional[str]:
        """
        Try multiple extraction methods in sequence
        
        Available methods:
        - 'text_search': Search through raw text
        - 'coordinates': Use coordinate-based extraction  
        - 'ocr': Convert to image and use OCR
        - 'pattern': Use regex pattern matching
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
                result = method_map[method_name](page_no, table_identifier, column_name, row_name)
                
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
    
    def debug_page_content(self, page_no: int):
        """Debug method to see page content structure"""
        page = self.doc[page_no - 1]
        
        print(f"=== DEBUG PAGE {page_no} CONTENT ===")
        
        # Method 1: Show raw text
        print("\n--- RAW TEXT ---")
        text = page.get_text()
        lines = [line.strip() for line in text.split('\n') if line.strip()]
        for i, line in enumerate(lines[:20]):  # First 20 lines
            print(f"{i:2d}: {line}")
        
        # Method 2: Show text blocks with coordinates
        print("\n--- TEXT BLOCKS WITH COORDINATES ---")
        text_dict = page.get_text("dict")
        block_count = 0
        
        for block in text_dict["blocks"]:
            if "lines" in block and block_count < 10:  # First 10 blocks
                print(f"\nBlock {block_count}:")
                for line in block["lines"]:
                    for span in line["spans"]:
                        if span["text"].strip():
                            bbox = span["bbox"]
                            print(f"  [{bbox[0]:.1f}, {bbox[1]:.1f}] '{span['text'].strip()}'")
                block_count += 1
        
        # Method 3: Try to find potential tables
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
    extractor = AlternativePDFExtractor(pdf_path)
    
    try:
        # Show debug info first
        if config.get('debug', False):
            extractor.debug_page_content(config['page_no'])
        
        # Try extraction
        result = extractor.extract_data(
            page_no=config['page_no'],
            table_identifier=config['table_identifier'],
            column_name=config['column_name'],
            row_name=config['row_name'],
            methods=config.get('methods', None)
        )
        
        return result
        
    finally:
        extractor.close()

# Example usage
if __name__ == "__main__":
    # Configuration
    config = {
        'page_no': 1,
        'table_identifier': 'Financial Summary',
        'column_name': 'Q1 2024', 
        'row_name': 'Total Revenue',
        'methods': ['text_search', 'line_parsing', 'pattern'],  # Specify methods to try
        'debug': True  # Set to True to see page structure
    }
    
    # result = extract_with_debug('sample.pdf', config)
    # print(f"Final result: {result}")
