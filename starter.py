import os.path
from llama_index import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
)

documents_dir = "./modulhandbuch"

# create new index
documents = SimpleDirectoryReader(documents_dir).load_data()
index = VectorStoreIndex.from_documents(documents)
# store it for later
index.storage_context.persist(persist_dir=documents_dir)
print("Index created and stored for later use.")

# check if storage already exists
if  os.path.exists(documents_dir) and not any(file.endswith('.json') for file in os.listdir(documents_dir)):
    print("documents/docstore.json not found. Creating index...")
    # load the documents and create the index
    documents = SimpleDirectoryReader(documents_dir).load_data()
    index = VectorStoreIndex.from_documents(documents)
    # store it for later
    index.storage_context.persist(persist_dir=documents_dir)
    print("Index created and stored for later use.")
#
#else:
#    # load the existing index
#    storage_context = StorageContext.from_defaults(persist_dir=documents_dir)
#    index = load_index_from_storage(storage_context)

