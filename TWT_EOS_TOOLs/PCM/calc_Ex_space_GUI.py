import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.integrate import quad
import math
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.font_manager as fm

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False

class BeamFocusCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("带状束流磁场计算器")
        self.root.geometry("900x700")
        self.root.configure(bg='#f0f0f0')
        
        # 物理常量 (固化在程序中)
        self.CONSTANTS = {
            'yita': 1.76e11,      # 迁移率，单位 m^2/(V·s)
            'erb': 8.85e-12,      # 真空介电常数，单位 F/m
        }
        
        # 创建界面
        self.create_widgets()
        
        # 结果存储
        self.results = []
        
    def create_widgets(self):
        # 主框架
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 标题
        title_label = ttk.Label(main_frame, text="带状束流横向聚焦磁场计算器", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # 输入参数框架
        input_frame = ttk.LabelFrame(main_frame, text="输入参数", padding="10")
        input_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 10))
        
        # 输入字段
        self.create_input_fields(input_frame)
        
        # 计算按钮
        calc_button = ttk.Button(main_frame, text="计算磁场", command=self.calculate, 
                                style='Accent.TButton')
        calc_button.grid(row=2, column=0, pady=10, sticky=(tk.W, tk.E))
        
        # 清除按钮
        clear_button = ttk.Button(main_frame, text="清除结果", command=self.clear_results)
        clear_button.grid(row=2, column=1, pady=10, padx=(10, 0), sticky=(tk.W, tk.E))
        
        # 结果显示框架
        result_frame = ttk.LabelFrame(main_frame, text="计算结果", padding="10")
        result_frame.grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=(10, 0))
        
        # 创建结果显示区域
        self.create_result_display(result_frame)
        
        # 配置网格权重
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(3, weight=1)
        result_frame.columnconfigure(0, weight=1)
        result_frame.rowconfigure(1, weight=1)
        
    def create_input_fields(self, parent):
        # 定义输入字段
        fields = [
            ("电压 U (V):", "U", "23000.0"),
            ("电流 I (A):", "I", "0.45"),
            ("通道高度 H (mm):", "H", "0.15"),
            ("带状束流厚度 b (mm):", "b", "0.12"),
            ("带状束流长宽比K_ab 起始值:", "K_start", "6"),
            ("带状束流长宽比K_ab 结束值:", "K_end", "8"),
            ("积分项数 n_terms:", "n_terms", "10")
        ]
        
        self.entries = {}
        
        for i, (label_text, key, default_value) in enumerate(fields):
            row = i // 2
            col = (i % 2) * 2
            
            label = ttk.Label(parent, text=label_text)
            label.grid(row=row, column=col, sticky=tk.W, padx=(0, 5), pady=2)
            
            entry = ttk.Entry(parent, width=12)
            entry.insert(0, default_value)
            entry.grid(row=row, column=col+1, sticky=(tk.W, tk.E), padx=(0, 20), pady=2)
            
            self.entries[key] = entry
            
        # 配置列权重
        parent.columnconfigure(1, weight=1)
        parent.columnconfigure(3, weight=1)
        
    def create_result_display(self, parent):
        # 结果文本框
        self.result_text = tk.Text(parent, height=8, width=50, font=('Consolas', 10))
        scrollbar_y = ttk.Scrollbar(parent, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar_x = ttk.Scrollbar(parent, orient=tk.HORIZONTAL, command=self.result_text.xview)
        
        self.result_text.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        self.result_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        
        # 图表框架
        self.figure, self.ax = plt.subplots(figsize=(6, 4))
        self.canvas = FigureCanvasTkAgg(self.figure, parent)
        self.canvas.get_tk_widget().grid(row=2, column=0, columnspan=2, pady=(10, 0), 
                                        sticky=(tk.W, tk.E, tk.N, tk.S))
        
        parent.rowconfigure(2, weight=1)
        
    def get_input_values(self):
        """获取并验证输入值"""
        try:
            values = {}
            
            # 基本参数
            values['U'] = float(self.entries['U'].get())
            values['I'] = float(self.entries['I'].get())
            values['H'] = float(self.entries['H'].get()) * 1e-3  # mm转m
            values['b'] = float(self.entries['b'].get()) * 1e-3  # mm转m
            
            # 计算参数
            values['K_start'] = int(self.entries['K_start'].get())
            values['K_end'] = int(self.entries['K_end'].get())
            values['n_terms'] = int(self.entries['n_terms'].get())
            
            # 验证输入范围
            if values['U'] <= 0 or values['I'] <= 0:
                raise ValueError("电压和电流必须为正值")
            if values['H'] <= 0 or values['b'] <= 0:
                raise ValueError("几何参数必须为正值")
            if values['K_start'] >= values['K_end']:
                raise ValueError("K_ab起始值必须小于结束值")
            if values['n_terms'] <= 0:
                raise ValueError("积分项数必须为正整数")
                
            return values
            
        except ValueError as e:
            messagebox.showerror("输入错误", f"输入参数有误: {str(e)}")
            return None
            
    def integrand_Ia(self, t, rx_val, ry_val, n_val):
        """被积函数"""
        t_sq_norm = (t**2) / (rx_val**2)
        yb = ry_val * np.sqrt(1.0 - t_sq_norm)
        freq = (2 * n_val + 1) * math.pi
        return np.sin(freq * yb) * np.exp(freq * (t - rx_val))
    
    def calculate_magnetic_field(self, values):
        """计算磁场的核心算法"""
        # 计算基础参数
        yita = self.CONSTANTS['yita']
        erb = self.CONSTANTS['erb']
        
        Vb = np.sqrt(2 * yita * values['U'])  # 基底速度
        
        results = []
        
        for K_ab in range(values['K_start'], values['K_end'] + 1):
            a = K_ab * values['b']  # 带状束流宽度
            rx = a / values['H']    # 归一化长度
            ry = values['b'] / values['H']  # 归一化厚度
            
            S = math.pi * a * values['b'] / 4  # 束流横截面积
            rho = values['I'] / (S * Vb)       # 电荷密度
            
            sum_total = 0.0
            
            for n in range(values['n_terms']):
                # 计算Ia的积分
                try:
                    Ia_val, _ = quad(self.integrand_Ia, -rx, rx, args=(rx, ry, n))
                    Ib_val = 0.0  # X0到rx积分区间为0
                    
                    # 累加电场分量
                    E_x = (2 * (Ia_val - Ib_val) / ((2*n + 1) * math.pi) * 
                           (rho * values['H']) / erb)
                    sum_total += E_x
                    
                except Exception as e:
                    messagebox.showwarning("计算警告", f"积分计算异常: {str(e)}")
                    continue
            
            # 计算磁场
            By = sum_total / Vb
            By_Gs = By * 1e4  # 特斯拉转高斯
            
            results.append({
                'K_ab': K_ab,
                'a': a * 1000,  # 转换为mm
                'By_T': By,
                'By_Gs': By_Gs,
                'rx': rx,
                'ry': ry,
                'rho': rho,
                'S': S * 1e6  # 转换为mm²
            })
            
        return results
    
    def calculate(self):
        """主计算函数"""
        values = self.get_input_values()
        if values is None:
            return
            
        try:
            # 显示计算中状态
            self.result_text.delete(1.0, tk.END)
            self.result_text.insert(tk.END, "正在计算，请稍候...\n")
            self.root.update()
            
            # 执行计算
            self.results = self.calculate_magnetic_field(values)
            
            # 显示结果
            self.display_results(values)
            self.plot_results()
            
        except Exception as e:
            messagebox.showerror("计算错误", f"计算过程中发生错误: {str(e)}")
            
    def display_results(self, input_values):
        """显示计算结果"""
        self.result_text.delete(1.0, tk.END)
        
        # 输入参数摘要
        self.result_text.insert(tk.END, "=== 计算参数摘要 ===\n")
        self.result_text.insert(tk.END, f"电压: {input_values['U']} V\n")
        self.result_text.insert(tk.END, f"电流: {input_values['I']} A\n")
        self.result_text.insert(tk.END, f"通道高度: {input_values['H']*1000:.2f} mm\n")
        self.result_text.insert(tk.END, f"束流厚度: {input_values['b']*1000:.2f} mm\n")
        self.result_text.insert(tk.END, f"积分项数: {input_values['n_terms']}\n\n")
        
        # 计算结果
        self.result_text.insert(tk.END, "=== 磁场计算结果 ===\n")
        self.result_text.insert(tk.END, f"{'K_ab':<4} {'宽度(mm)':<8} {'磁场(T)':<12} {'磁场(Gs)':<10} {'横截面积(mm²)':<12}\n")
        self.result_text.insert(tk.END, "-" * 60 + "\n")
        
        for result in self.results:
            self.result_text.insert(tk.END, 
                f"{result['K_ab']:<4} {result['a']:<8.3f} {result['By_T']:<12.6e} "
                f"{result['By_Gs']:<10.6f} {result['S']:<12.6f}\n")
        
        self.result_text.insert(tk.END, "\n=== 物理常数 ===\n")
        self.result_text.insert(tk.END, f"迁移率: {self.CONSTANTS['yita']:.2e} m²/(V·s)\n")
        self.result_text.insert(tk.END, f"真空介电常数: {self.CONSTANTS['erb']:.2e} F/m\n")
        
    def plot_results(self):
        """绘制结果图表"""
        if not self.results:
            return
            
        self.ax.clear()
        
        K_values = [r['K_ab'] for r in self.results]
        By_Gs_values = [r['By_Gs'] for r in self.results]
        
        self.ax.plot(K_values, By_Gs_values, 'bo-', linewidth=2, markersize=8)
        self.ax.set_xlabel('K_ab', fontsize=12)
        self.ax.set_ylabel('磁场强度 (Gs)', fontsize=12)
        self.ax.set_title('聚焦带状束流所需磁场', fontsize=14, fontweight='bold')
        self.ax.grid(True, alpha=0.3)
        
        # 添加数值标签
        for k, by in zip(K_values, By_Gs_values):
            self.ax.annotate(f'{by:.3f}', (k, by), textcoords="offset points", 
                           xytext=(0,10), ha='center', fontsize=10)
        
        self.ax.set_xticks(K_values)
        self.figure.tight_layout()
        self.canvas.draw()
        
    def clear_results(self):
        """清除结果"""
        self.result_text.delete(1.0, tk.END)
        self.results = []
        self.ax.clear()
        self.ax.set_title('计算结果图表')
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = BeamFocusCalculator(root)
    
    # 设置样式
    style = ttk.Style()
    style.theme_use('clam')
    
    # 运行应用
    root.mainloop()

if __name__ == "__main__":
    main()