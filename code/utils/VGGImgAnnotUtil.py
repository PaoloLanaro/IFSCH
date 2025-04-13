import json
from typing import Dict, List, Any

class VIAAnnotationParser:
    def __init__(self, json_path: str = None, json_data: Dict = None):
        """ Initialize the parser with either a JSON file path or direct JSON data.
        Takes in:
            json_path: Path to the VIA JSON file
            json_data: Direct JSON data (alternative to file path)
        """
        if json_path:
            with open(json_path, 'r') as f:
                self.data = json.load(f)
        elif json_data:
            self.data = json_data
        else:
            raise ValueError("Either json_path or json_data must be provided")

        self._validate_data()

    def _validate_data(self):
        """ Basic validation of the VIA JSON structure """
        if '_via_img_metadata' not in self.data:
            raise ValueError("Invalid VIA JSON format: '_via_img_metadata' missing")

    def get_image_metadata(self) -> Dict[str, Any]:
        """ Get all image metadata """
        return self.data['_via_img_metadata']

    def get_image_ids(self) -> List[str]:
        """ Get list of all image IDs in the annotation file """
        return list(self.data['_via_img_metadata'].keys())

    def get_image_info(self, image_id: str) -> Dict[str, Any]:
        """ Get complete information for a specific image.
        Takes in image_id: The ID of the image (key in _via_img_metadata) """
        return self.data['_via_img_metadata'][image_id]

    def get_filename(self, image_id: str) -> str:
        """ Get filename for a specific image ID """
        return self.data['_via_img_metadata'][image_id]['filename']

    def get_image_size(self, image_id: str) -> int:
        """ Get file size in bytes for a specific image ID """
        return self.data['_via_img_metadata'][image_id]['size']

    def get_regions(self, image_id: str) -> List[Dict]:
        """ Get all regions (annotations) for a specific image ID """
        return self.data['_via_img_metadata'][image_id].get('regions', [])

    def get_shape_types(self, image_id: str) -> List[str]:
        """ Get list of shape types used in annotations for an image """
        regions = self.get_regions(image_id)
        return [r['shape_attributes']['name'] for r in regions]

    def get_polygons(self, image_id: str) -> List[Dict]:
        """ Get all polygon annotations for a specific image ID """
        regions = self.get_regions(image_id)
        return [r for r in regions if r['shape_attributes']['name'] == 'polygon']

    def get_region_attributes(self, image_id: str) -> List[Dict]:
        """ Get all region attributes for a specific image ID """
        regions = self.get_regions(image_id)
        return [r['region_attributes'] for r in regions]

    def get_project_name(self) -> str:
        """ Get the project name if available """
        return self.data.get('_via_settings', {}).get('project', {}).get('name', '')

    def count_annotations(self) -> Dict[str, int]:
        """ Count annotations per image """
        return {
            img_id: len(img_data.get('regions', []))
            for img_id, img_data in self.data['_via_img_metadata'].items()
        }

# Example usage:
if __name__ == "__main__":
    # Example with direct JSON data (like your excerpt)
    example_data = {
        "_via_settings": {...},  # your settings data
        "_via_img_metadata": {...}  # your metadata
    }

    # Initialize parser
    parser = VIAAnnotationParser(json_data=example_data)

    # Get all image IDs
    image_ids = parser.get_image_ids()
    print(f"Image IDs: {image_ids}")

    # Get information about the first image
    if image_ids:
        first_image = image_ids[0]
        print(f"\nFilename: {parser.get_filename(first_image)}")
        print(f"Size: {parser.get_image_size(first_image)} bytes")
        print(f"Annotation count: {len(parser.get_regions(first_image))}")

        # Print polygon coordinates for each region
        for i, region in enumerate(parser.get_regions(first_image)):
            if region['shape_attributes']['name'] == 'polygon':
                print(f"\nRegion {i + 1} (polygon):")
                print(f"X coordinates: {region['shape_attributes']['all_points_x']}")
                print(f"Y coordinates: {region['shape_attributes']['all_points_y']}")
                print(f"Attributes: {region['region_attributes']}")
