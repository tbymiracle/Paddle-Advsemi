from torch.utils import data
from dataset.voc_dataset import VOCDataSet, VOCGTDataSet

train_dataset = VOCDataSet(args.data_dir, args.data_list, crop_size=input_size,
                           scale=args.random_scale, mirror=args.random_mirror, mean=IMG_MEAN)

train_ids = range(10000)
partial_size = 5000
train_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])
train_remain_sampler = data.sampler.SubsetRandomSampler(train_ids[partial_size:])
train_gt_sampler = data.sampler.SubsetRandomSampler(train_ids[:partial_size])

trainloader = data.DataLoader(train_dataset,
                batch_size=10, sampler=train_sampler, num_workers=3, pin_memory=True)
trainloader_remain = data.DataLoader(train_dataset,
                batch_size=10, sampler=train_remain_sampler, num_workers=3, pin_memory=True)
trainloader_gt = data.DataLoader(train_gt_dataset,
                batch_size=10, sampler=train_gt_sampler, num_workers=3, pin_memory=True)
print(train_sampler)