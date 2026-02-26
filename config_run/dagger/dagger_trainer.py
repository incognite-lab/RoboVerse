# config_run/dagger/dagger_trainer.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
from loguru import logger as log

class DAggerBuffer:
    def __init__(self, max_size, device):
        self.max_size = max_size # Maximální počet snímků celkem
        self.device = device
        # Předalokované tensory jsou lepší než list.append, ale pro jednoduchost necháme list
        # Pokud by to padalo, přepíšeme na Circular Buffer s předalokací
        self.imgs = []
        self.expert_actions = []
        self.ptr = 0
        self.current_size = 0

    def add_batch(self, batch_imgs_uint8, batch_actions):
        """
        Přijímá [Num_Envs, C, H, W] a [Num_Envs, Actions]
        """
        # Pokud máme 9000 prostředí, nemůžeme uložit všechno (to by byl buffer plný hned).
        # Vybereme náhodných např. 100 vzorků z aktuálního kroku.
        # To zajistí diverzitu a nepřehltí paměť.

        num_envs = batch_imgs_uint8.shape[0]
        # Uložíme jen zlomek dat (např. 100 náhodných robotů z 9000)
        indices = torch.randint(0, num_envs, (200,), device=self.device)

        selected_imgs = batch_imgs_uint8[indices]
        selected_actions = batch_actions[indices]

        # Přidání do listu (jednoduchá implementace)
        self.imgs.append(selected_imgs)
        self.expert_actions.append(selected_actions)

        # Ořezání, pokud je toho moc
        # (Tato implementace je trochu "humpolácká", ale funkční pro začátek)
        if len(self.imgs) > 1000: # Max 1000 batchů po 200 vzorcích = 200 000 fotek
             self.imgs.pop(0)
             self.expert_actions.pop(0)

    def sample(self, batch_size):
        # Spojíme vše, co máme
        if not self.imgs:
            return None, None

        all_imgs = torch.cat(self.imgs, dim=0)
        all_actions = torch.cat(self.expert_actions, dim=0)

        # Ochrana, kdyby buffer byl menší než batch
        valid_batch_size = min(batch_size, all_imgs.shape[0])

        indices = torch.randint(0, all_imgs.shape[0], (valid_batch_size,), device=self.device)

        # Převod na float až tady
        return all_imgs[indices].float() / 255.0, all_actions[indices]

def train_dagger_step(student_net, optimizer, buffer, batch_size=1024):
    if len(buffer.imgs) == 0:
        return 0.0

    student_net.train()
    imgs, expert_acts = buffer.sample(batch_size)

    # 1. Predikce studenta
    student_acts = student_net(imgs)

    # 2. Výpočet chyby (Supervised Learning - MSE)
    loss_fn = nn.MSELoss()
    loss = loss_fn(student_acts, expert_acts)

    # 3. Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
