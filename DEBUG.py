
def cout(name, val):
    print('------------------DEBUG【', name, '】 = ',val)


def coutobj(obj):
    print('------------------DEBUG---------------')
    attributes = dir(obj)

    # 过滤掉不需要的内置属性
    attributes = [attr for attr in attributes if not attr.startswith('__')]

    # 获取并打印每个属性的值
    for attr in attributes:
        value = getattr(obj, attr)
        print(f'{attr}: {value}')
