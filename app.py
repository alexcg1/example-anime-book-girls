from docarray import Document, DocumentArray
import torch
import torchvision


def preproc(d: Document):
    return (
        d.load_uri_to_image_tensor()
        .set_image_tensor_shape((80, 60))
        .set_image_tensor_normalization()
        .set_image_tensor_channel_axis(-1, 0)
    )


def reverse_preproc(d: Document):
    return (d.set_image_tensor_channel_axis(0, -1)).set_image_tensor_inv_normalization()


data_dir = "Anime-Girls-Holding-Programming-Books"
max_files = 100

docs = DocumentArray.from_files(
    f"{data_dir}/**/*.png", f"{data_dir}/**/*.jpg", size=max_files
)

docs.apply(preproc)

model = torchvision.models.resnet50(pretrained=True)
model.fx = torch.nn.Identity()

docs.embed(model, device="cpu")

# Query image
query_doc = Document(
    uri=f"{data_dir}/Python/Aoba_Suzukaze_techgo_Python_For_Beginners.png"
)
query_doc.load_uri_to_image_tensor

query_doc = preproc(query_doc)
query_doc.embed(model, device="cpu")

matches = query_doc.match(docs).matches

for match in matches:
    print(match.uri)

matches.apply(reverse_preproc)
