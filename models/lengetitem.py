# 现在我们实例化一个类，让它的一个属性是一个数组，然后通过调用__getitem__方法来索引属性中数组的值
# 开始
class Data1:
    def __init__(self, contents):
        self.content = contents
    
    def __len__(self):
        return 1

    def __getitem__(self, index):
        print('调用了__getitem__()函数')
        return self.content[index]

b = Data1([1, 2, 3, 4])

for index, content in enumerate(b):
    print(index)
    print(content)
