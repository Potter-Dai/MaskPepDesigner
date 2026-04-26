# test dataset
# dataset = DiffoldDataset(
#     config=config,
#     mode="train",
#     unifold_db_path="/mnt/data/projects/unifold/data_0916/traineval",
#     uf_distill_db_path="/mnt/data/projects/unifold/data_0916/sd",
#     esmatlas_db_path="/mnt/data/projects/unifold/esmatlas/hq/lmdbs",
#     seed=0,
# )
# samples = [dataset[i] for i in range(100)]
# uids = [db.decompress_uid(s["uid"]) for s in samples]
# recur_print(samples[0])
# print(uids[:20])