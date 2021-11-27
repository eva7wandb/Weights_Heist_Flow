from torchvision.transforms.functional import to_pil_image
from torchcam.methods import SmoothGradCAMpp
from torchcam.utils import overlay_mask
from torchsummary import summary
import matplotlib.pyplot as plt


def show_model_summary(model, input_size=(3, 32, 32)):
    summary(model, input_size=input_size)


def visualize_gradcam(raw_image, processed_img, model, layer_name):
    cam_extractor = SmoothGradCAMpp(model.eval(), layer_name)
    out = model(processed_img.unsqueeze(0))
    activation_map = cam_extractor(out.squeeze(0).argmax().item(), out)
    result = overlay_mask(
        to_pil_image(raw_image), 
        to_pil_image(activation_map[0], mode='F'), 
        alpha=0.5
    )
    
    return raw_image.permute(1, 2, 0).cpu(), activation_map[0].cpu().numpy(), result

    
def plot_probabilites(pred, actual_class, pred_class, class_names):
    prob_bar = plt.bar(range(len(pred)), pred)
    prob_bar[actual_class].set_color('g' if pred_class==actual_class else 'r')
    plt.xticks(range(10), [x.upper() for x in class_names], rotation=90)
    plt.yticks([])
    plt.show()
    

def visualize_sample(trainer, sample, cam_layer_name='layer4'):
    raw_img, act_map, overlay = visualize_gradcam(
        sample['data_unnormalized'], sample['data'], trainer.net.cpu(), cam_layer_name
    )
    
    plt.figure(figsize=(10, 5))
    plt.subplot(221)
    plt.imshow(raw_img); plt.title('raw_image'); plt.axis('off')
    plt.subplot(222)
    plt.imshow(act_map); plt.title('activation\nmap'); plt.axis('off')
    plt.subplot(223)
    plt.imshow(overlay); plt.title('overlay'); plt.axis('off')
    
    pred, actual_class, pred_class, class_names = (
        sample['scores'], 
        sample['actual_class'], 
        sample['pred_class'], 
        trainer.test_loader.dataset.classes
    )

    plt.subplot(224)
    prob_bar = plt.bar(range(len(pred)), pred)
    prob_bar[pred_class].set_color('r')
    prob_bar[actual_class].set_color('g')
    plt.xticks(range(10), [x.upper() for x in class_names], rotation=90)
    plt.title('score')

def visualize_loss(logs):
    loss_train = [x['train_loss'] for x in logs]
    loss_val = [x['test_loss'] for x in logs]
    epochs = range(1,len(logs)+1)
    plt.plot(epochs, loss_train, 'g', label='Training loss')
    plt.plot(epochs, loss_val, 'b', label='Testing loss')
    plt.title('Training and Testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()