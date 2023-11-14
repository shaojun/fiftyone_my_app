import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.brain as fob
name = "electric_bicycle"
dataset_dir = "/home/shao/Downloads/test/electric_bicycle"
# Create the dataset
dataset = fo.Dataset.from_dir(
    dataset_dir=dataset_dir,
    dataset_type=fo.types.ImageDirectory,
    name=name,
)

# View summary info about the dataset
print(dataset)

# Print the first few samples in the dataset
print(dataset.head())

fob.compute_uniqueness(dataset)

print(dataset)
# View a sample from the dataset
print(dataset.first())

# Launch the app
session = fo.launch_app(dataset, port=5151)
# Show least unique images first
least_unique_view = dataset.sort_by("uniqueness", reverse=False)

# Open view in App
session.view = least_unique_view

session.wait()
# Get currently selected images from App
dup_ids = session.selected
print(dup_ids)

# Get view containing selected samples
dups_view = dataset.select(dup_ids)

# Mark as duplicates
for sample in dups_view:
    sample.tags.append("duplicate")
    sample.save()

# Select samples with `duplicate` tag
dups_tag_view = dataset.match_tags("duplicate")

# Open view in App
session.view = dups_tag_view

from fiftyone import ViewField as F
import os
import shutil

# Get samples that do not have the `duplicate` tag
no_dups_view = dataset.match(~F("tags").contains("duplicate"))

export_full_path = os.path.join(dataset_dir,'no_duplicated')
if os.path.exists(export_full_path):
    shutil.rmtree(export_full_path)
# Export dataset to disk as a classification directory tree
no_dups_view.export(
    export_full_path,
    dataset_type=fo.types.ImageDirectory,
)
