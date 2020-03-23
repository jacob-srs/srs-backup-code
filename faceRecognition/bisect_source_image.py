#
# 作者：lee
# 时间： 20200321
# 描述： 将原始图像集分类成注册图像集与测试图像集
# 参考：TestAccuracy.py
#


import argparse
import os
import sys


class Source_2_Regist_Test():
    """ 将原始图像二分
    
    设计：原始图像主目录 xxx 下的子目录结构不变，但是每一个子目录中的图像选取两个
    成为注册图像，余下三个成为测试图像。生成的两个结果主目录分别命名为 xxx_regist
    与 xxx_test

    功能1： 命令行能接收原始图像目录路径
    """
    
    def __init__(self, parser_args=False, command_tuple=None):
        
        self.source_path = ""  # 字符串声明
        
        # 对命令行模式或调试模式进行判断，对命令行参数进行判断
        if command_tuple is not None:  # 调试模式
            self.__sys_argv = command_tuple
            if parser_args is False:  # parser_arg == False，输入三个地址
                self.__check_all_paths()
            elif parser_args is True:  # 输入 source 地址
                self.__check_source_path()            
        else:
            self.__sys_argv = self.__read_command_args_use_sys()
            if parser_args is False:  # parser_arg == False，输入三个地址
                self.__check_all_paths()
            elif parser_args is True:  # 输入 source 地址
                self.__check_source_path()

    def __read_command_args_use_sys(self):
        """ 读取命令行参数，使用 sys.argv """
        
        return sys.argv  # 命令行参数列表
    
    def __read_command_args_use_getopt(self):
        """ TODO 读取命令行参数，使用 getopt """
        
        pass

    def __check_source_path(self):  # 目前主要用这个函数
        """ TODO 接收 source 路径并将 register 与 test 保存在于 source 所在目录中
        
        由于时间有限，"--source_only" 的位置写死，
        所以 argv[0] = 文件.py，argv[1] = --source_only， argv[2] = 路径
        """
        print("in check source path")
        
        source_path = ""  # 声明空字符串
        if len(self.__sys_argv) < 3:  # 没有输入 source 路径
            print("\n..Error::source path is not entered, exiting...")
            sys.exit(0)
        
        if not os.path.exists(self.__sys_argv[2]):
            print("\n..Error::source path is not  exist, exiting...")
            sys.exit(0)

        source_path = self.__sys_argv[2]
        self.source_path = source_path
        return source_path

    def __make_dir(self, dir_name):
        """ 实例私有函数，判断并创建目录 """
        
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)
        else:
            pass

    def create_dir_register_test(self):
        """ TODO 创建 register, test 文件夹
        
        注1：尝试 os.getcwd()
        注2：source，register，test都在同级目录下
        """
        
        print("in create_dir_register_test")      

        # 创建 source 同级分类目录 register 与 test
        root_path = "\\".join(self.source_path.split("\\")[:-1])
        source_name = self.source_path.split("\\")[-1]
        register_dir_name = source_name + "_register"
        test_dir_name = source_name + "_test"

        register_dir_path = os.path.join(root_path, register_dir_name)
        test_dir_path = os.path.join(root_path, test_dir_name)

        self.__make_dir(register_dir_path)
        self.__make_dir(test_dir_path)
       
        # 在分类目录中循环复制 source 中的文件结构， glob？
        source_sub_dirs = os.listdir((self.source_path))
        for name_sub_dir in source_sub_dirs:
            register_sub_dir = os.path.join(register_dir_path, name_sub_dir)
            test_sub_dir = os.path.join(test_dir_path, name_sub_dir)

            self.__make_dir(register_sub_dir)
            self.__make_dir(test_sub_dir)
    
    def __check_all_paths(self):
        """ 读取命令行输入的路径 """
        
        # argv = self.__sys_argv
        if len(self.__sys_argv) < 3:
            print("\n..Error::enter maximum 3 path variables in order::[source][register][test]")
            sys.exit(0)

        source_not_exist_flag = False
        path_list = {}   # 将三个路径使用字典保存
        for i, path in enumerate(self.__sys_argv):
            if i == 0:  # 跳过命令行 文件名
                continue            
            
            if not os.path.exists(path):
                print("\n..Warning::path arg \"%s\" in pos [%s] is not exist..." %(path, i))
                if i == 1:  # source 路径不存在，退出                    
                    source_not_exist_flag = True
                    continue
        if source_not_exist_flag is True:
            print("\n..Error::source path is not exist, exiting...")
            sys.exit(0)


    def get_command_args(self):
        """ 获取命令行参数 """
        
        return self.__sys_argv

    def create_directory(self, dir_name):
        """ TODO 创建文件目录 """
        
        pass


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(
        description="\n..divide source to register and test...")
    parser.add_argument(
        "--source_only", type=str, help="enter only source path", 
        default=None)
    
    args = parser.parse_args()
    parser_flag = False

    command_tuple = ()
    args.source_only = True
    if args.source_only:
        parser_flag = True
        command_tuple = ("hello.py", "--source_only", r"D:\P_project_face_recognition\casia_set")
    else:
        command_tuple = (
            "hello.py",
            r"D:\P_project_face_recognition\casia_set",
            r"D:\P_project_face_recognition\hello",
            r"D:\P_project_face_recognition\there")
    
    s2rt = Source_2_Regist_Test(parser_flag, command_tuple)
    argv = s2rt.get_command_args()
    print("\n参数列表::%s，参数数目::%s" %(str(argv), len(argv)))

    s2rt.create_dir_register_test()
