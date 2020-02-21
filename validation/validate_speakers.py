import os
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
# from tsnecuda import TSNE


def plot_tsne(args, embedding, labels, epoch, step):
    fp = os.path.join(args.out_dir, 'tsne', '{}-{}.png'.format(epoch, step))
    if not os.path.exists(os.path.dirname(fp)):
        os.makedirs(os.path.dirname(fp))

    figure = plt.figure(figsize=(8,8), dpi=120)
    plt.scatter(embedding[:, 0], embedding[:, 1], c=labels.ravel())
    plt.axis('off')
    plt.savefig(fp, bbox_inches='tight')
    return figure

def tsne(args, features, labels):
    features = features.reshape(features.size(0) * features.size(1), -1).cpu()
    embedding = TSNE().fit_transform(features)
    labels = labels.reshape(-1, 1).cpu().numpy()
    return embedding, labels

def validate_speakers(args, dataset, model, optimizer, epoch, step, writer):

    max_speakers = 10
    batch_size = 40
    input_size = (args.batch_size, 1, 20480)

    model.eval()
    with torch.no_grad():
        latent_rep_size, latent_rep_len = model.module.get_latent_size(input_size)
        features = torch.zeros(max_speakers, batch_size, latent_rep_size * latent_rep_len).to(args.device)
        labels = torch.zeros(max_speakers, batch_size).to(args.device)

        for idx, speaker_idx in enumerate(dataset.speaker_dict):
            if idx == 10:
                break

            model_in = dataset.get_audio_by_speaker(speaker_idx, batch_size=batch_size)
            model_in = model_in.to(args.device)
            z, c = model.module.get_latent_representations(model_in)
            
            z_repr = z.permute(0, 2, 1)
            c_repr = c.permute(0, 2, 1)

            features[idx, :, :] = c_repr.reshape((batch_size, -1))
            labels[idx, :] = idx

    embedding, labels = tsne(args, features, labels)
    figure = plot_tsne(args, embedding, labels, epoch, step)

    # add to TensorBoard
    writer.add_embedding(embedding, metadata=labels, global_step=step)
    writer.add_figure('TSNE', figure, global_step=step)
    writer.flush()
    model.train()
