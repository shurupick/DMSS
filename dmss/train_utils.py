import torch
from tqdm import tqdm


def save_model(model, path):
    pass


def train_and_evaluate_one_epoch(
    model, train_dataloader, valid_dataloader, optimizer, scheduler, loss_fn, device
):
    # Set the model to training mode
    model.train()
    train_loss = 0
    for batch in tqdm(train_dataloader, desc="Training"):
        images, masks = batch
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        outputs = model(images)

        loss = loss_fn(outputs, masks)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

    scheduler.step()
    avg_train_loss = train_loss / len(train_dataloader)

    # Set the model to evaluation mode
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in tqdm(valid_dataloader, desc="Evaluating"):
            images, masks = batch
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, masks)

            val_loss += loss.item()

    avg_val_loss = val_loss / len(valid_dataloader)
    return avg_train_loss, avg_val_loss


def train_model(
    model,
    train_dataloader,
    valid_dataloader,
    optimizer,
    scheduler,
    loss_fn,
    device,
    epochs,
):
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        avg_train_loss, avg_val_loss = train_and_evaluate_one_epoch(
            model,
            train_dataloader,
            valid_dataloader,
            optimizer,
            scheduler,
            loss_fn,
            device,
        )
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)

        logging.info(
            f"Epoch {epoch + 1}/{epochs}, Training Loss: {avg_train_loss:.2f}, Validation Loss: {avg_val_loss:.2f}"
        )

    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
    }
    return history
