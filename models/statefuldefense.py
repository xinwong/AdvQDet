import numpy as np
import torch
import torchvision.transforms.functional as F
import torchvision
import torchvision.transforms as transforms
from models.resnet import resnet152, resnet20, resnet101, vit16
# from models.iot_sqa import IOTSQAClassifier, IOTSQAEncoder
from abc import abstractmethod
from multiprocessing import Pool
import hashlib
from collections import Counter
from skimage.feature import local_binary_pattern
import cv2
import clip
from acpt.resnet_multi_bn import resnet18, proj_head

def convert_models_to_fp32(model):
    for p in model.parameters():
        p.data = p.data.float()
        if p.grad:
            p.grad.data = p.grad.data.float()

def multiGPU_CLIP(sim_encoder, images, prompt_token):
    if prompt_token is not None:
        bs = images.size(0)
        prompt_token = prompt_token.repeat(bs, 1, 1)

    img_embed = sim_encoder(images, prompt_token)

    return img_embed

class StateModule:
    @abstractmethod
    def getDigest(self, img):
        """
        Returns a digest of the image
        :param img:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def resultsTopk(self, img, k):
        """
        Return a list of top k tuples (distance, prediction) - smallest distance first
        :param img:
        :param k:
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def resetCache(self):
        """
        Reset the cache
        :return:
        """
        raise NotImplementedError

    @abstractmethod
    def add(self, img, prediction):
        """
        Add an image to the cache
        :param img:
        :return:
        """
        raise NotImplementedError


class BlackLight(StateModule):
    def __init__(self, arguments):
        self.window_size = arguments["window_size"]
        self.num_hashes_keep = arguments["num_hashes_keep"]
        self.round = arguments["round"]
        self.step_size = arguments["step_size"]
        self.input_shape = arguments["input_shape"]
        self.salt_type = arguments["salt"]
        if self.salt_type:
            self.salt = (np.random.rand(*tuple(self.input_shape)) * 255).astype(np.int16)
        else:
            self.salt = (np.zeros(tuple(self.input_shape)) * 255).astype(np.int16)

        self.cache = {}
        self.inverse_cache = {}
        self.input_idx = 0
        self.pool = Pool(processes=arguments["num_processes"])

    @staticmethod
    def hash_helper(arguments):
        img = arguments['img']
        idx = arguments['idx']
        window_size = arguments['window_size']
        return hashlib.sha256(img[idx:idx + window_size]).hexdigest()

    def preprocess(self, array, salt, round=1, normalized=True):
        if len(array.shape) != 3:
            raise Exception("expected 3d image")
        if (normalized):
            # input image normalized to [0,1]
            array = np.array(array.cpu()) * 255.
        array = (array + salt) % 255.
        array = array.reshape(-1)

        array = np.around(array / round, decimals=0) * round
        array = array.astype(np.int16)
        return array

    def getDigest(self, img):
        img = self.preprocess(img, self.salt, self.round)
        total_len = int(len(img))
        idx_ls = []

        for el in range(int((total_len - self.window_size + 1) / self.step_size)):
            idx_ls.append({"idx": el * self.step_size, "img": img, "window_size": self.window_size})
        hash_list = self.pool.map(BlackLight.hash_helper, idx_ls)
        hash_list = list(set(hash_list))
        hash_list = [r[::-1] for r in hash_list]
        hash_list.sort(reverse=True)
        return hash_list[:self.num_hashes_keep]

    def resetCache(self):
        self.cache = {}
        self.inverse_cache = {}
        self.input_idx = 0
        if self.salt_type:
            self.salt = (np.random.rand(*tuple(self.input_shape)) * 255).astype(np.int16)
        else:
            self.salt = (np.zeros(tuple(self.input_shape)) * 255).astype(np.int16)

    def add(self, img, prediction):
        self.input_idx += 1
        hashes = self.getDigest(img)
        for el in hashes:
            if el not in self.inverse_cache:
                self.inverse_cache[el] = [self.input_idx]
            else:
                self.inverse_cache[el].append(self.input_idx)
        self.cache[self.input_idx] = prediction

    def resultsTopk(self, img, k):
        hashes = self.getDigest(img)
        sets = list(map(self.inverse_cache.get, hashes))
        sets = [i for i in sets if i is not None]
        sets = [item for sublist in sets for item in sublist]
        if not sets:
            return []
        sets = Counter(sets)
        result = [((self.num_hashes_keep - x[1]) / self.num_hashes_keep, self.cache[x[0]]) for x in sets.most_common(k)]
        return result

class TokenPrompter(torch.nn.Module):
    def __init__(self, prompt_len) -> None:
        super(TokenPrompter, self).__init__()

        self.prompt = torch.nn.Parameter(torch.randn([1, prompt_len, 768]))
    
    def forward(self):
        return self.prompt

class AdvQDet(StateModule):
    def __init__(self, arguments):
        add_prompt_len=20

        checkpoint = torch.load('/path/to/checkpoint.pth.tar')
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, self.preprocess = clip.load("ViT-B/32", device=self.device, prompt_len=add_prompt_len)
 
        convert_models_to_fp32(model)
        model = torch.nn.DataParallel(model)

        model.eval()
        self.encoder = model.module.encode_image
    
        add_prompter = TokenPrompter(prompt_len=add_prompt_len)
        add_prompter = torch.nn.DataParallel(add_prompter).to(self.device)

        add_prompter.load_state_dict(checkpoint['add_prompter'])
        add_prompter.eval()
        self.ind_prompt = add_prompter()

        model_dtype = next(model.parameters()).dtype
        print(f"Model dtype: {model_dtype}")
        add_prompter_dtype = next(add_prompter.parameters()).dtype
        print(f"Add_prompter dtype: {add_prompter_dtype}")
        
        self.input_shape = arguments["input_shape"]

        self.cache = {}


    def getDigest(self, img):
        if len(img.shape) != 3:
            raise Exception("expected 3d image")

        pil_img = transforms.ToPILImage()(img)
        image = self.preprocess(pil_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embed = multiGPU_CLIP(self.encoder, image, self.ind_prompt).squeeze(0)

        return embed

    def resetCache(self):
        self.cache = {}

    def add(self, img, prediction):
        if len(img.shape) != 3:
            raise Exception("expected 3d image")
        encoding = self.getDigest(img)
        self.cache[encoding] = prediction
    
    def resultsTopk(self, img, k):
        img = torch.clamp(img, 0, 1)
        embed = self.getDigest(img)
        dists = []
        preds = []
        for query_embed, pred in self.cache.items():
            dist = torch.cosine_similarity(embed, query_embed, dim=0).item()
            dists.append(dist)
            preds.append(pred)
        top_dists = np.argsort(dists)
        result = [(dists[i], preds[i]) for i in top_dists][::-1]

        return result


class OSDEncoder(torch.nn.Module):
    def __init__(self):
        super(OSDEncoder, self).__init__()

        self.conv1 = torch.nn.Conv2d(3, 32, 3, padding='same')
        self.conv2 = torch.nn.Conv2d(32, 32, 3)
        self.drop1 = torch.nn.Dropout2d(0.25)

        self.conv3 = torch.nn.Conv2d(32, 64, 3, padding='same')
        self.conv4 = torch.nn.Conv2d(64, 64, 3)
        self.drop2 = torch.nn.Dropout2d(0.25)

        self.fc1 = torch.nn.Linear(64 * 6 * 6, 512)
        self.drop3 = torch.nn.Dropout2d(0.5)
        self.fc2 = torch.nn.Linear(512, 256)

    def forward(self, x):
        x = torch.nn.functional.relu(self.conv1(x))
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv2(x)), 2)
        x = self.drop1(x)
        x = torch.nn.functional.relu(self.conv3(x))
        x = torch.nn.functional.max_pool2d(torch.nn.functional.relu(self.conv4(x)), 2)
        x = self.drop2(x)
        x = x.permute(0, 2, 3, 1)
        x = x.reshape(-1, 64 * 6 * 6)
        x = torch.nn.functional.relu(self.fc1(x))
        x = self.drop3(x)
        x = self.fc2(x)
        return x


class OriginalStatefulDetector(StateModule):
    def __init__(self, arguments):
        self.encoder = OSDEncoder().cuda()
        checkpoint = torch.load(arguments["encoder_path"])
        self.encoder.load_state_dict(checkpoint)
        self.encoder.eval()
        self.input_shape = arguments["input_shape"]
        if arguments["salt"] is not None:
            self.salt = arguments["salt"]
        else:
            self.salt = np.zeros(self.input_shape).astype(np.int16)

        self.cache = {}

    def getDigest(self, img):
        if len(img.shape) != 3:
            raise Exception("expected 3d image")
        return self.encoder(img.to(next(self.encoder.parameters()).device).detach().unsqueeze(0)).squeeze(0)

    def resetCache(self):
        self.cache = {}

    def add(self, img, prediction):
        if len(img.shape) != 3:
            raise Exception("expected 3d image")
        encoding = self.getDigest(img)
        self.cache[encoding] = prediction

    def resultsTopk(self, img, k):
        # print(torch.min(img), torch.max(img))
        img = torch.clamp(img, 0, 1)
        embed = self.getDigest(img)
        dists = []
        preds = []
        for query_embed, pred in self.cache.items():
            dist = torch.linalg.norm(embed - query_embed).item()
            dists.append(dist)
            preds.append(pred)
        top_dists = np.argsort(dists)[:k]
        # top_dists = np.argpartition(dists, k - 1)
        result = [(dists[i], preds[i]) for i in top_dists]
        return result


class PIHA(StateModule):
    def __init__(self, arguments):
        super(PIHA, self).__init__()
        self.input_shape = arguments["input_shape"]
        self.block_size = arguments["block_size"]
        self.cache_predictions = {}
        self.input_idx = 0
        self.cache = self.getDigest(torch.zeros(arguments["input_shape"]))

    def _piha_hash(self, x):
        N = x.shape[2]
        # Image preprocessing
        x = x.cpu().numpy().transpose(1, 2, 0)
        x_filtered = cv2.GaussianBlur(x, (3, 3), 1)

        # Color space transformation
        x_hsv = cv2.cvtColor(x_filtered, cv2.COLOR_RGB2HSV)

        # Use only H channel for HSV color space
        x_h = x_hsv[:, :, 0].reshape((N, N, 1))
        x_h = np.pad(x_h,
                     ((0, self.block_size - N % self.block_size), (0, self.block_size - N % self.block_size), (0, 0)),
                     'constant')
        N = x_h.shape[0]

        # Block division and feature matrix calculation
        blocks_h = [x_h[i:i + self.block_size, j:j + self.block_size] for i in range(0, N, self.block_size) for j in
                    range(0, N, self.block_size)]
        features_h = np.array([np.sum(block) for block in blocks_h]).reshape(
            (N // self.block_size, N // self.block_size))

        # Local binary pattern feature extraction
        features_lbp = local_binary_pattern(features_h, 8, 1)

        # Hash generation
        # hash_array = ''.join([f'{(int(_)):x}' for _ in features_lbp.flatten().tolist()])
        # hash_array = ''.join([format(int(_), '02x') for _ in features_lbp.flatten().tolist()])
        hash_array = features_lbp.flatten()
        hash_array = np.expand_dims(hash_array, axis=0)
        return hash_array

    def getDigest(self, img):
        h = self._piha_hash(img)
        return h

    def resetCache(self):
        self.cache = self.getDigest(torch.zeros(tuple(self.input_shape)))

    def add(self, img, prediction):
        self.input_idx += 1
        hash = self.getDigest(img)
        self.cache = np.concatenate((self.cache, hash))
        self.cache_predictions[self.input_idx] = prediction

    def resultsTopk(self, img, k):
        hash = self.getDigest(img)
        hamming_dists = np.count_nonzero(hash != self.cache, axis=1) / self.cache.shape[1]
        closest = np.argsort(hamming_dists)[:k]
        # remove dummy element if present
        closest = closest[closest != 0]
        if len(closest) == 0:
            return []
        result = [(hamming_dists[i], self.cache_predictions[i]) for i in closest]
        return result


class NoOpState(StateModule):
    def __init__(self, arguments):
        pass

    def getDigest(self, img):
        return img

    def resetCache(self):
        pass

    def add(self, img, prediction):
        pass

    def resultsTopk(self, img, k):
        return []


class StatefulClassifier(torch.nn.Module):
    def __init__(self, model, state_module, hyperparameters):
        super().__init__()
        self.config = hyperparameters
        self.model = model
        self.state_module = state_module
        self.threshold = hyperparameters["threshold"]
        self.aggregation = hyperparameters["aggregation"]
        self.add_cache_hit = hyperparameters["add_cache_hit"]
        self.reset_cache_on_hit = hyperparameters["reset_cache_on_hit"]
        self.cache_hits = 0
        self.total = 0
        self.distances = []
        self.similar_count = 0
        self.similar_num = 0
        self.count = 0
        self.gate = True
        self.five_shot=0
        self.process = []
   
    def reset(self):
        self.state_module.resetCache()
        self.cache_hits = 0
        self.total = 0
        self.distances = []
        self.similar_count = 0
        self.gate = True

    def forward_single(self, x):
        self.total += 1

        cached_prediction = None
        similar = False
        if self.aggregation == 'closest':
            similarity_result = self.state_module.resultsTopk(x, 1)
            if len(similarity_result) > 0:
                dist, cached_prediction = similarity_result[0]
                self.distances.append(dist)
                if dist <= self.threshold:
                    if self.add_cache_hit:
                        self.state_module.add(x, cached_prediction)
                    self.cache_hits += 1
                    if self.reset_cache_on_hit:
                        self.state_module.resetCache()
                    similar = True

        elif self.aggregation == 'average':
            similarity_result = self.state_module.resultsTopk(x, self.config['num_to_average'])
            if len(similarity_result) >= self.config['num_to_average']:
                dist, cached_prediction = similarity_result[0]
                dists = [dist for (dist, _) in similarity_result]
                if np.mean(dists) <= self.threshold:
                    if self.add_cache_hit:
                        self.state_module.add(x, cached_prediction)
                    self.cache_hits += 1
                    if self.reset_cache_on_hit:
                        self.state_module.resetCache()
                    similar = True

        elif self.aggregation == 'cosine':
            similarity_result = self.state_module.resultsTopk(x, 1)
            if len(similarity_result) > 0:
                dist, cached_prediction = similarity_result[0]
                # print("max distance: {}".format(dist))
                self.distances.append(dist)
                # print([dist for (dist, _) in similarity_result])
                if dist >= self.threshold:
                    if self.add_cache_hit:
                        self.state_module.add(x, cached_prediction)
                    self.cache_hits += 1
                    if self.reset_cache_on_hit:
                        self.state_module.resetCache()
                    similar = True

        if similar:
            self.similar_count+=1

            if self.total <= 4 and self.gate:
                self.five_shot += 1
                self.gate = False
                # print("Few-Shot Detect Rate:{}/100".format(self.five_shot))

            if self.similar_count <= 3:
                # print("detect similar image for the {} time during the {}th run".format(self.similar_count, self.total))
                if self.similar_count == 1:
                    self.similar_num += self.total
                    self.count += 1
                    print("Avg detect num: {}, detect rate: {}/100".format(self.similar_num / self.count, self.count))

            if self.config["action"] != 'rejection_silent':
                # cached_prediction = -1 * torch.ones_like(cached_prediction)
                return cached_prediction.cuda(), True

            prediction = self.model(x.to(next(self.model.parameters()).device).unsqueeze(0)).detach()
            return prediction, False

        if self.total <= 4 and self.gate:
            # print("Few-Shot Detect Rate:{}/100".format(self.five_shot))
        prediction = self.model(x.to(next(self.model.parameters()).device).unsqueeze(0)).detach()
        self.state_module.add(x, prediction.detach().cpu())
        return prediction, False

    def forward_batch(self, x):
        batch_size = x.shape[0]
        logits, is_cache = [], []
        for i in range(batch_size):
            pred, is_cached = self.forward_single(x[i])
            logits.append(pred)
            is_cache.append(is_cached)
        logits = torch.cat(logits, dim=0)
        probs = torch.nn.functional.softmax(logits, dim=1)
        return probs, is_cache

    def forward(self, x):
        if len(x.shape) == 3:
            return self.forward_single(x)
        else:
            return self.forward_batch(x)


def init_stateful_classifier(config):
    if config['architecture'] == 'resnet20':
        model = torch.load("models/pretrained/resnet20-12fca82f-single.pth", map_location="cpu")
        model.eval()
    elif config['architecture'] == 'resnet34':
        model = torch.load("models/pretrained/resnet34_gtsrb.pth")
        model.eval()
    elif config['architecture'] == 'resnet101':
        model = resnet101()
        model.eval()  
    elif config['architecture'] == 'resnet152':
        model = resnet152()
        model.eval()
    elif config['architecture'] == 'ViT-S/16':
        model = vit16()
        model.eval()
    elif config['architecture'] == 'celebahq':
        class CelebAHQClassifier(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.model = torchvision.models.resnet18(pretrained=True)
                num_features = self.model.fc.in_features
                self.model.fc = torch.nn.Linear(num_features, 307)
                self.model.load_state_dict(torch.load(
                    "models/pretrained/facial_identity_classification_transfer_learning_with_ResNet18_resolution_256.pth"))
                self.xform = torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

            def forward(self, x):
                x = self.xform(x)
                return self.model(x)

        model = CelebAHQClassifier()
        model.eval()
    else:
        raise NotImplementedError("Architecture not supported.")

    if config["state"]["type"] == "blacklight":
        state_module = BlackLight(config["state"])
    elif config["state"]["type"] == "PIHA":
        state_module = PIHA(config["state"])
    elif config["state"]["type"] == "OSD":
        state_module = OriginalStatefulDetector(config["state"])
    elif config["state"]["type"] == "AdvQDet":
        state_module = AdvQDet(config["state"])
    elif config["state"]["type"] == "no_op":
        state_module = NoOpState(config["state"])
    else:
        raise NotImplementedError("State module not supported.")

    return StatefulClassifier(model, state_module, config)
