import os
import file_tools as ft

class SaveManager:
    def __init__(self, model_dir):
        self.weights_dir = os.path.join(model_dir, "weights")
    def cleanup(self):
        return 0
    def iter_weights(self):
        for file in ft.scandir(self.weights_dir):
            epoch_num = int(file.name.split('_')[0])
            yield epoch_num, file
    def get_epoch_nums(self):
        epoch_nums = []
        for epoch_num, _ in self.iter_weights():
            if epoch_num not in epoch_nums:
                epoch_nums.append(epoch_num)
        return sorted(epoch_nums)


class RemoveOldestManager(SaveManager):
    def __init__(self, model_dir, keep_latest=5):
        super().__init__(model_dir)
        self.keep_latest = keep_latest
    def cleanup(self):
        deleted = 0
        epoch_nums = self.get_epoch_nums()
        if len(epoch_nums) <= self.keep_latest:
            return deleted
        remove_epochs = epoch_nums[:-self.keep_latest]
        for epoch_num, file in self.iter_weights():
            if epoch_num in remove_epochs:
                deleted += file.size()
                file.delete()
        return deleted


class RemoveIntermediateManager(SaveManager):
    def __init__(
            self, model_dir, trigger_size, save_rate,
            cleanup_mod=4, mod_update=lambda mod: max(mod // 2, 1)):
        super().__init__(model_dir)
        self.trigger_size = trigger_size
        self.save_rate = save_rate
        self.cleanup_mod = cleanup_mod
        self.mod_update = mod_update
    def size(self):
        folder_path = self.weights_dir
        return ft.calc_folder_sizes(folder_path)[folder_path]
    def over_capacity(self):
        return self.size() > self.trigger_size
    def delete_epoch(self, epoch_num, epoch_nums, cleanup_mod):
        return epoch_num in epoch_nums and (epoch_num % (self.save_rate * cleanup_mod) == 0)
    def cleanup(self, max_iters=None):
        deleted = 0
        if not self.over_capacity():
            return deleted
        epoch_nums = self.get_epoch_nums()[1:-1]
        cleanup_mod = self.cleanup_mod
        # print("epoch_nums:", epoch_nums)
        iters = 0
        while epoch_nums and self.over_capacity():
            removed = set()
            for epoch_num, file in self.iter_weights():
                if not self.delete_epoch(epoch_num, epoch_nums, cleanup_mod):
                    continue
                deleted += file.size()
                removed.add(epoch_num)
                file.delete()
            for rm_epoch in removed:
                epoch_nums.remove(rm_epoch)
            cleanup_mod = self.mod_update(cleanup_mod)
            iters += 1
            # print("size:", self.size() / (2 ** 30) - deleted)
            # print("capacity:", self.trigger_size / (2 ** 30))
            # print("new_delete_mod:", delete_mod)
            if max_iters and iters >= max_iters:
                break
        return deleted


class KeepIntermediateManager(RemoveIntermediateManager):
    def __init__(
            self, model_dir, trigger_size, save_rate,
            cleanup_mod=1, mod_update=lambda mod: mod * 2):
        super().__init__(
            model_dir, trigger_size, save_rate,
            cleanup_mod=cleanup_mod, mod_update=mod_update
        )
    def delete_epoch(self, epoch_num, epoch_nums, cleanup_mod):
        return epoch_num in epoch_nums and (epoch_num % (self.save_rate * cleanup_mod))
