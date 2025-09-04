import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import customtkinter as ctk
import sympy as sp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
import json
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'SimSun', 'FangSong']
plt.rcParams['axes.unicode_minus'] = False

# 设置customtkinter主题
ctk.set_appearance_mode("light")
ctk.set_default_color_theme("blue")

class MagneticFieldAnalyzer(ctk.CTk):
    def __init__(self):
        super().__init__()
        
        # 窗口设置
        self.title("磁场周期长度分析器 v2.0")
        self.geometry("1400x900")
        self.minsize(1200, 800)
        
        # 初始化参数
        self.init_parameters()
        
        # 创建界面
        self.create_widgets()
        
        # 初始计算
        self.calculate_results()
    
    def init_parameters(self):
        """初始化计算参数"""
        self.params = {
            'U': ctk.DoubleVar(value=25000),
            'I': ctk.DoubleVar(value=0.45),
            'yita': ctk.DoubleVar(value=1.76e11),
            'erb': ctk.DoubleVar(value=8.85e-12),
            't': ctk.DoubleVar(value=0.12e-3),
            'w_factor': ctk.DoubleVar(value=6)
        }
        
        self.plot_range = {
            'B_min': ctk.DoubleVar(value=-2),
            'B_max': ctk.DoubleVar(value=2),
            'L_min': ctk.DoubleVar(value=0.1),
            'L_max': ctk.DoubleVar(value=5)
        }
        
        self.results = {}
        
        # 绑定变量更新事件
        for var in self.params.values():
            var.trace('w', self.on_param_change)
        for var in self.plot_range.values():
            var.trace('w', self.on_range_change)
    
    def create_widgets(self):
        """创建主界面"""
        # 主框架
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # 左侧参数面板
        self.create_parameter_panel()
        
        # 右侧主面板
        self.create_main_panel()
        
        # 状态栏
        self.create_status_bar()
    
    def create_parameter_panel(self):
        """创建参数输入面板"""
        param_frame = ctk.CTkFrame(self, width=350)
        param_frame.grid(row=0, column=0, padx=(10, 5), pady=10, sticky="nsew")
        param_frame.grid_propagate(False)
        
        # 标题
        title_label = ctk.CTkLabel(param_frame, text="输入参数", 
                                 font=ctk.CTkFont(size=18, weight="bold"))
        title_label.pack(pady=(20, 10))
        
        # 创建滚动框架
        scrollable_frame = ctk.CTkScrollableFrame(param_frame)
        scrollable_frame.pack(fill="both", expand=True, padx=20, pady=(0, 20))
        
        # 参数输入区域
        params_info = [
            ('U', '电压 (V)', 'U'),
            ('I', '电流 (A)', 'I'),
            ('yita', 'η (C/kg)', 'yita'),
            ('erb', 'ε₀ (F/m)', 'erb'),
            ('t', '厚度 t (m)', 't'),
            ('w_factor', '宽度系数 (w = 系数×t)', 'w_factor')
        ]
        
        self.param_entries = {}
        for i, (key, label, var_key) in enumerate(params_info):
            frame = ctk.CTkFrame(scrollable_frame)
            frame.pack(fill="x", pady=5)
            
            label_widget = ctk.CTkLabel(frame, text=label, font=ctk.CTkFont(size=12))
            label_widget.pack(anchor="w", padx=10, pady=(10, 5))
            
            entry = ctk.CTkEntry(frame, textvariable=self.params[var_key], 
                               font=ctk.CTkFont(size=11))
            entry.pack(fill="x", padx=10, pady=(0, 10))
            self.param_entries[key] = entry
        
        # 绘图范围设置
        range_frame = ctk.CTkFrame(scrollable_frame)
        range_frame.pack(fill="x", pady=(20, 5))
        
        range_title = ctk.CTkLabel(range_frame, text="绘图范围", 
                                 font=ctk.CTkFont(size=14, weight="bold"))
        range_title.pack(pady=(10, 5))
        
        # B范围
        b_frame = ctk.CTkFrame(range_frame)
        b_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(b_frame, text="B 范围 (T):", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=5, pady=2)
        b_range_frame = ctk.CTkFrame(b_frame)
        b_range_frame.pack(fill="x", padx=5, pady=(0, 5))
        
        ctk.CTkEntry(b_range_frame, textvariable=self.plot_range['B_min'], 
                    width=80, font=ctk.CTkFont(size=10)).pack(side="left", padx=2)
        ctk.CTkLabel(b_range_frame, text="到", font=ctk.CTkFont(size=10)).pack(side="left", padx=2)
        ctk.CTkEntry(b_range_frame, textvariable=self.plot_range['B_max'], 
                    width=80, font=ctk.CTkFont(size=10)).pack(side="left", padx=2)
        
        # L范围
        l_frame = ctk.CTkFrame(range_frame)
        l_frame.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(l_frame, text="L 范围 (cm):", font=ctk.CTkFont(size=11)).pack(anchor="w", padx=5, pady=2)
        l_range_frame = ctk.CTkFrame(l_frame)
        l_range_frame.pack(fill="x", padx=5, pady=(0, 5))
        
        ctk.CTkEntry(l_range_frame, textvariable=self.plot_range['L_min'], 
                    width=80, font=ctk.CTkFont(size=10)).pack(side="left", padx=2)
        ctk.CTkLabel(l_range_frame, text="到", font=ctk.CTkFont(size=10)).pack(side="left", padx=2)
        ctk.CTkEntry(l_range_frame, textvariable=self.plot_range['L_max'], 
                    width=80, font=ctk.CTkFont(size=10)).pack(side="left", padx=2)
        
        # 按钮区域
        button_frame = ctk.CTkFrame(scrollable_frame)
        button_frame.pack(fill="x", pady=(20, 10))
        
        calc_button = ctk.CTkButton(button_frame, text="重新计算", 
                                  command=self.calculate_results,
                                  font=ctk.CTkFont(size=12, weight="bold"))
        calc_button.pack(pady=10)
        
        export_button = ctk.CTkButton(button_frame, text="导出结果", 
                                    command=self.export_results,
                                    font=ctk.CTkFont(size=12))
        export_button.pack(pady=(0, 10))
        
        load_button = ctk.CTkButton(button_frame, text="加载参数", 
                                  command=self.load_parameters,
                                  font=ctk.CTkFont(size=12))
        load_button.pack(pady=(0, 10))
    
    def create_main_panel(self):
        """创建主显示面板"""
        main_frame = ctk.CTkFrame(self)
        main_frame.grid(row=0, column=1, padx=(5, 10), pady=10, sticky="nsew")
        main_frame.grid_columnconfigure(0, weight=1)
        main_frame.grid_rowconfigure(1, weight=1)
        
        # 结果显示面板
        self.create_results_panel(main_frame)
        
        # 图表显示面板
        self.create_plot_panel(main_frame)
    
    def create_results_panel(self, parent):
        """创建结果显示面板"""
        results_frame = ctk.CTkFrame(parent, height=200)
        results_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        results_frame.grid_propagate(False)
        
        title_label = ctk.CTkLabel(results_frame, text="计算结果", 
                                 font=ctk.CTkFont(size=18, weight="bold"))
        title_label.pack(pady=(15, 10))
        
        # 结果显示区域
        results_container = ctk.CTkFrame(results_frame)
        results_container.pack(fill="both", expand=True, padx=15, pady=(0, 15))
        
        # 创建结果显示网格
        self.result_labels = {}
        result_items = [
            ('Bbri', '布里渊磁场', 'T'),
            ('Bo', '最小峰值磁场', 'T'),
            ('recommended_charge', '推荐充磁', 'T'),
            ('recommended_period', '推荐周期', 'mm'),
            ('max_period', '最大周期', 'mm')
        ]
        
        for i, (key, label, unit) in enumerate(result_items):
            row = i // 3
            col = i % 3
            
            item_frame = ctk.CTkFrame(results_container)
            item_frame.grid(row=row, column=col, padx=5, pady=5, sticky="ew")
            results_container.grid_columnconfigure(col, weight=1)
            
            name_label = ctk.CTkLabel(item_frame, text=label, 
                                    font=ctk.CTkFont(size=11, weight="bold"))
            name_label.pack(pady=(8, 2))
            
            value_label = ctk.CTkLabel(item_frame, text="计算中...", 
                                     font=ctk.CTkFont(size=14))
            value_label.pack(pady=(0, 8))
            
            self.result_labels[key] = (value_label, unit)
    
    def create_plot_panel(self, parent):
        """创建图表显示面板"""
        plot_frame = ctk.CTkFrame(parent)
        plot_frame.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="nsew")
        plot_frame.grid_columnconfigure(0, weight=1)
        plot_frame.grid_rowconfigure(1, weight=1)
        
        plot_title = ctk.CTkLabel(plot_frame, text="磁场 vs 周期长度关系图", 
                                font=ctk.CTkFont(size=16, weight="bold"))
        plot_title.grid(row=0, column=0, pady=(15, 5))
        
        # 创建matplotlib图表
        self.fig = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        
        # 创建画布
        canvas_frame = ctk.CTkFrame(plot_frame)
        canvas_frame.grid(row=1, column=0, padx=15, pady=(5, 15), sticky="nsew")
        canvas_frame.grid_columnconfigure(0, weight=1)
        canvas_frame.grid_rowconfigure(0, weight=1)
        
        self.canvas = FigureCanvasTkAgg(self.fig, canvas_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        
        # 添加工具栏
        toolbar_frame = ctk.CTkFrame(canvas_frame)
        toolbar_frame.grid(row=1, column=0, sticky="ew", pady=(5, 0))
        
        save_plot_btn = ctk.CTkButton(toolbar_frame, text="保存图表", 
                                    command=self.save_plot, width=100)
        save_plot_btn.pack(side="left", padx=5, pady=5)
        
        refresh_plot_btn = ctk.CTkButton(toolbar_frame, text="刷新图表", 
                                       command=self.update_plot, width=100)
        refresh_plot_btn.pack(side="left", padx=5, pady=5)
    
    def create_status_bar(self):
        """创建状态栏"""
        self.status_var = tk.StringVar()
        self.status_var.set("就绪")
        
        status_frame = ctk.CTkFrame(self, height=30)
        status_frame.grid(row=1, column=0, columnspan=2, sticky="ew", padx=10, pady=(0, 10))
        status_frame.grid_propagate(False)
        
        status_label = ctk.CTkLabel(status_frame, textvariable=self.status_var, 
                                  font=ctk.CTkFont(size=11))
        status_label.pack(side="left", padx=10, pady=5)
    
    def on_param_change(self, *args):
        """参数变化时的回调"""
        self.after(500, self.calculate_results)  # 延迟计算，避免频繁更新
    
    def on_range_change(self, *args):
        """绘图范围变化时的回调"""
        self.after(500, self.update_plot)
    
    def calculate_results(self):
        """计算结果"""
        try:
            self.status_var.set("计算中...")
            self.update()
            
            # 获取参数值
            U = self.params['U'].get()
            I = self.params['I'].get()
            yita = self.params['yita'].get()
            erb = self.params['erb'].get()
            t = self.params['t'].get()
            w_factor = self.params['w_factor'].get()
            
            # 计算派生参数
            w = w_factor * t
            S = w * t * np.pi / 4
            Vb = np.sqrt(2 * yita * U)
            p = I / (S * Vb)
            
            # 计算关键磁场值
            Bbri = np.sqrt(1.414 * I / (w * t * erb * yita**1.5 * U**0.5))
            Bo = 2 * Bbri
            recommended_charge = Bo / 0.5
            
            # 计算推荐周期
            Wp = np.sqrt(yita * I / (S * Vb * erb))
            recommended_period = 2 * np.pi * Vb / Wp / 3 * 1e3  # 转换为毫米
            
            # 计算最大周期
            WoS = Bo * yita
            k = 2 * np.pi / (sp.symbols('L') * 1e-2)  # L单位转换为cm
            fun5 = (WoS**2/(Vb*k)**2)*3.1/4 - (yita*p)/((Vb*k)**2*erb) - 1

            try:
                # 尝试求解方程
                Lp = sp.solve(fun5, sp.symbols('L'))
                if Lp:
                    # 取第一个实数解
                    Lp_val = abs(float(Lp[0])) * 10  # 单位转换
                    max_period = Lp_val / 2
                else:
                    max_period = float('nan')
            except:
                max_period = float('nan')
            
            # 存储结果
            self.results = {
                'Bbri': Bbri,
                'Bo': Bo,
                'recommended_charge': recommended_charge,
                'recommended_period': recommended_period,
                'max_period': max_period
            }
            
            # 更新显示
            self.update_results_display()
            self.update_plot()
            
            self.status_var.set(f"计算完成 - {datetime.now().strftime('%H:%M:%S')}")
            
        except Exception as e:
            messagebox.showerror("计算错误", f"计算过程中发生错误：{str(e)}")
            self.status_var.set("计算错误")
    
    def update_results_display(self):
        """更新结果显示"""
        for key, (label, unit) in self.result_labels.items():
            if key in self.results:
                value = self.results[key]
                if abs(value) >= 1e3 or abs(value) <= 1e-3:
                    text = f"{value:.3e} {unit}"
                else:
                    text = f"{value:.6f} {unit}"
                label.configure(text=text)
    
    def update_plot(self):
        """更新图表"""
        try:
            self.ax.clear()
            
            # 获取参数
            U = self.params['U'].get()
            I = self.params['I'].get()
            yita = self.params['yita'].get()
            erb = self.params['erb'].get()
            t = self.params['t'].get()
            w_factor = self.params['w_factor'].get()
            
            w = w_factor * t
            S = w * t * np.pi / 4
            Vb = np.sqrt(2 * yita * U)
            p = I / (S * Vb)
            
            # 获取绘图范围
            B_min = self.plot_range['B_min'].get()
            B_max = self.plot_range['B_max'].get()
            L_min = self.plot_range['L_min'].get()
            L_max = self.plot_range['L_max'].get()
            
            # 创建网格
            B_vals = np.linspace(B_min, B_max, 100)
            L_vals = np.linspace(L_min, L_max, 100)
            B_grid, L_grid = np.meshgrid(B_vals, L_vals)
            
            # 计算各方程的值
            Wo_grid = yita * B_grid
            k_grid = 2 * np.pi / (L_grid * 1e-2)
            
            # 避免除零
            k_grid = np.where(k_grid == 0, 1e-10, k_grid)
            
            Z1 = (Wo_grid / (Vb * k_grid))**2 - 3.64
            Z2 = (Wo_grid / (Vb * k_grid))**2 * 0.5 - (yita * p) / ((Vb * k_grid)**2 * erb) * (w / (w + t))
            Z3 = Z2 - 1
            Z4 = (Wo_grid / (Vb * k_grid))**2 * 3.1/4 - (yita * p) / ((Vb * k_grid)**2 * erb) - 1
            Z6 = Wo_grid - 8 * Vb / (5 * t)
            
            # 绘制等高线
            colors = ['red', 'olive', 'purple', 'blue', 'green']
            labels = ['Z1', 'Z2', 'Z3', 'Z4', 'Z6']
            
            for Z, color, label in zip([Z1, Z2, Z3, Z4, Z6], colors, labels):
                self.ax.contour(B_grid, L_grid, Z, levels=[0], colors=color, linewidths=2)
            
            # 设置图表属性
            self.ax.set_xlabel('B (T)', fontsize=12)
            self.ax.set_ylabel('L (cm)', fontsize=12)
            self.ax.set_title('磁场 vs 周期长度关系', fontsize=14, fontweight='bold')
            self.ax.grid(True, alpha=0.3)
            
            # 添加图例
            legend_elements = [Line2D([0], [0], color=color, lw=2, label=label) 
                             for color, label in zip(colors, labels)]
            self.ax.legend(handles=legend_elements, loc='best')
            
            # 设置坐标轴范围
            self.ax.set_xlim(B_min, B_max)
            self.ax.set_ylim(L_min, L_max)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            print(f"绘图错误: {e}")
    
    def save_plot(self):
        """保存图表"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf"), ("All files", "*.*")]
            )
            if filename:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("保存成功", f"图表已保存到: {filename}")
        except Exception as e:
            messagebox.showerror("保存错误", f"保存图表时发生错误：{str(e)}")
    
    def export_results(self):
        """导出计算结果"""
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("Text files", "*.txt"), ("All files", "*.*")]
            )
            if filename:
                export_data = {
                    "parameters": {key: var.get() for key, var in self.params.items()},
                    "plot_range": {key: var.get() for key, var in self.plot_range.items()},
                    "results": self.results,
                    "timestamp": datetime.now().isoformat(),
                    "version": "2.0"
                }
                
                with open(filename, 'w', encoding='utf-8') as f:
                    json.dump(export_data, f, indent=2, ensure_ascii=False)
                
                messagebox.showinfo("导出成功", f"结果已导出到: {filename}")
        except Exception as e:
            messagebox.showerror("导出错误", f"导出结果时发生错误：{str(e)}")
    
    def load_parameters(self):
        """加载参数"""
        try:
            filename = filedialog.askopenfilename(
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )
            if filename:
                with open(filename, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # 加载参数
                if "parameters" in data:
                    for key, value in data["parameters"].items():
                        if key in self.params:
                            self.params[key].set(value)
                
                # 加载绘图范围
                if "plot_range" in data:
                    for key, value in data["plot_range"].items():
                        if key in self.plot_range:
                            self.plot_range[key].set(value)
                
                messagebox.showinfo("加载成功", f"参数已从 {filename} 加载")
                self.calculate_results()
        except Exception as e:
            messagebox.showerror("加载错误", f"加载参数时发生错误：{str(e)}")


def main():
    """主函数"""
    try:
        app = MagneticFieldAnalyzer()
        app.mainloop()
    except Exception as e:
        print(f"程序启动错误: {e}")

if __name__ == "__main__":
    main()