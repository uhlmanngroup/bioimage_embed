def check_dataloader()
for i, (rgb, gt) in enumerate(dataloader):
    print(f"batch {i+1}:")
    # some plots
    # for i in range(bs):
    plt.figure(figsize=(10, 5))
    plt.subplot(221)
    plt.imshow(torch.squeeze(rgb[0, 0]))
    plt.title(f"RGB img{i+1}")
    plt.subplot(222)
    plt.imshow(torch.squeeze(gt[0, 0]))
    plt.title(f"GT img{i+1}")
    break
    # break
    # plt.show()