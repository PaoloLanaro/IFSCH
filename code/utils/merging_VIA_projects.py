import json
from collections import defaultdict

class VIAProjectMerger:
    def __init__(self, merge_strategy='skip'):
        """
        merge_strategy: How to handle duplicate image IDs
          'skip' - keep existing annotations (default)
          'overwrite' - replace with new annotations
          'merge' - combine regions from both files
        """
        self.merge_strategy = merge_strategy
        self.merged_data = None
        self.duplicates = defaultdict(list)

    def load_file(self, file_path):
        """Load a VIA JSON file"""
        with open(file_path, 'r') as f:
            return json.load(f)

    def merge(self, input_files, output_file):
        """
        Merge multiple VIA JSON files
        """
        # Initialize with first file
        self.merged_data = self.load_file(input_files[0])
        
        # Track all seen image IDs
        all_image_ids = set(self.merged_data['_via_img_metadata'].keys())

        # Process remaining files
        for file_path in input_files[1:]:
            file_data = self.load_file(file_path)
            
            for img_id, img_metadata in file_data['_via_img_metadata'].items():
                if img_id in all_image_ids:
                    self._handle_duplicate(img_id, img_metadata)
                else:
                    self._add_new_image(img_id, img_metadata)
                    all_image_ids.add(img_id)

        # Update project name
        self.merged_data['_via_settings']['project']['name'] = "merged_project"
        
        # Save merged file
        with open(output_file, 'w') as f:
            json.dump(self.merged_data, f, indent=2)
            
        return self.duplicates

    def _handle_duplicate(self, img_id, new_metadata):
        """Handle duplicate image entries based on merge strategy"""
        existing = self.merged_data['_via_img_metadata'][img_id]
        
        # Record duplicate occurrence
        self.duplicates[img_id].append({
            'existing': existing,
            'new': new_metadata
        })

        if self.merge_strategy == 'skip':
            return
            
        if self.merge_strategy == 'overwrite':
            self.merged_data['_via_img_metadata'][img_id] = new_metadata
        elif self.merge_strategy == 'merge':
            # Combine regions from both annotations
            existing_regions = existing.get('regions', [])
            new_regions = new_metadata.get('regions', [])
            self.merged_data['_via_img_metadata'][img_id]['regions'] = (
                existing_regions + new_regions
            )

    def _add_new_image(self, img_id, metadata):
        """Add new image entry to merged data"""
        self.merged_data['_via_img_metadata'][img_id] = metadata

# Usage Example
if __name__ == "__main__":
    merger = VIAProjectMerger(merge_strategy='merge')  # Choose your strategy
    
    input_files = [
        '../../data/.segmented_images/picture1_climbing.json',
        '../../data/.segmented_images/picture2_climbing.json',
        '../../data/.segmented_images/picture3_climbing.json',
        '../../data/.segmented_images/picture4_climbing.json',
    ]
    
    duplicates = merger.merge(
        input_files=input_files,
        output_file='merged_project.json'
    )
    
    print(f"Merged project created with {len(duplicates)} duplicate(s) handled")
