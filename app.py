import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Зареждане на данните
@st.cache_data
def load_data():
    Fi_data = pd.read_csv('data/Fi.csv')
    Fi_data.columns = ['y', 'x', 'Fi']  # Осигуряваме съвместимост на колоните
    H_data = pd.read_csv('data/H.csv')
    
    # Конвертиране към float за сигурност
    Fi_data['Fi'] = Fi_data['Fi'].astype(float)
    H_data['H'] = H_data['H'].astype(float)
    
    # Премахване на дублиращи се записи
    Fi_data = Fi_data.drop_duplicates(subset=['x', 'y', 'Fi'])
    return Fi_data, H_data

Fi_data, H_data = load_data()

# Подготовка на данните за Fi
fi_aggregated_groups = {}
fi_interpolators = {}
fi_values_available = sorted(Fi_data['Fi'].unique())

for fi in fi_values_available:
    group = Fi_data[Fi_data['Fi'] == fi].sort_values(by='x')
    fi_aggregated_groups[fi] = group
    
    x = group['x'].values
    y = group['y'].values
    
    if len(x) < 2:
        def constant_func(x_val, y_const=y[0]):
            return np.full_like(x_val, y_const)
        fi_interpolators[fi] = constant_func
    else:
        fi_interpolators[fi] = interp1d(x, y, kind='linear', bounds_error=False, fill_value="extrapolate")

# Подготовка на данните за H
h_values_available = sorted(H_data['H'].unique())
h_to_x = {}
h_to_y_range = {}

for h in h_values_available:
    h_group = H_data[H_data['H'] == h]
    h_to_x[h] = h_group['x'].mean()
    h_to_y_range[h] = (h_group['y'].min(), h_group['y'].max())

# Функции за интерполация
def interpolate_fi(fi_value):
    fi_value = float(fi_value)
    if fi_value in fi_values_available:
        return fi_interpolators[fi_value]
    
    lower_fi = max([f for f in fi_values_available if f <= fi_value], default=None)
    upper_fi = min([f for f in fi_values_available if f >= fi_value], default=None)
    
    if lower_fi is None and upper_fi is None:
        return None
    elif lower_fi is None:
        return fi_interpolators[upper_fi]
    elif upper_fi is None:
        return fi_interpolators[lower_fi]
    
    def interpolated_func(x):
        y_lower = fi_interpolators[lower_fi](x)
        y_upper = fi_interpolators[upper_fi](x)
        ratio = (fi_value - lower_fi) / (upper_fi - lower_fi)
        return y_lower + ratio * (y_upper - y_lower)
    
    return interpolated_func

def interpolate_h(h_value):
    h_value = float(h_value)
    if h_value in h_values_available:
        return h_to_x[h_value]
    
    lower_h = max([h for h in h_values_available if h <= h_value], default=None)
    upper_h = min([h for h in h_values_available if h >= h_value], default=None)
    
    if lower_h is None and upper_h is None:
        return None
    elif lower_h is None:
        return h_to_x[upper_h]
    elif upper_h is None:
        return h_to_x[lower_h]
    
    ratio = (h_value - lower_h) / (upper_h - lower_h)
    return h_to_x[lower_h] + ratio * (h_to_x[upper_h] - h_to_x[lower_h])

def find_closest_values(value, available_values):
    value = float(value)
    available_values = [float(v) for v in available_values]
    
    lower = max([v for v in available_values if v <= value], default=None)
    upper = min([v for v in available_values if v >= value], default=None)
    return lower, upper

# Streamlit UI
st.title('Номограма за активно напрежение на срязване τb')
st.subheader('BG Pavement Design Guide (2002) - Фиг.9.8')

# Определяне на обхватите
min_fi = float(min(fi_values_available))
max_fi = float(max(fi_values_available))
min_h = float(min(h_values_available))
max_h = float(max(h_values_available))

# Избор на стойности
col1, col2 = st.columns(2)
with col1:
    fi_value = st.number_input(
        'Стойност за Fi', 
        min_value=min_fi, 
        max_value=max_fi, 
        value=min_fi, 
        step=1.0,
        format="%.1f"
    )
with col2:
    h_value = st.number_input(
        'Стойност за H', 
        min_value=min_h, 
        max_value=max_h, 
        value=min_h, 
        step=10.0,
        format="%.1f"
    )

# Проверка на входните стойности
try:
    fi_value = float(fi_value)
    h_value = float(h_value)
except ValueError:
    st.error("Моля, въведете валидни числови стойности")
    st.stop()

if not (min_fi <= fi_value <= max_fi):
    st.error(f"Стойността за Fi трябва да е между {min_fi} и {max_fi}")
    st.stop()

if not (min_h <= h_value <= max_h):
    st.error(f"Стойността за H трябва да е между {min_h} и {max_h}")
    st.stop()

# Изчисляване на τb
f_fi = interpolate_fi(fi_value)
x_h = interpolate_h(h_value)

if f_fi is None or x_h is None:
    st.error("Грешка при интерполация. Моля, проверете входните стойности.")
    st.stop()

y_tau = float(f_fi(x_h))

# Намиране на най-близките стойности
closest_fi_lower, closest_fi_upper = find_closest_values(fi_value, fi_values_available)
closest_h_lower, closest_h_upper = find_closest_values(h_value, h_values_available)

# Визуализация
fig, ax = plt.subplots(figsize=(12, 9))

# Намиране на граници
x_min = min(Fi_data['x'].min(), H_data['x'].min()) - 0.001
x_max = max(Fi_data['x'].max(), H_data['x'].max()) + 0.001
y_min = min(Fi_data['y'].min(), H_data['y'].min()) - 0.001
y_max = max(Fi_data['y'].max(), H_data['y'].max()) + 0.001

# Рисуване на всички изолинии (светли)
for fi_val in fi_values_available:
    group = fi_aggregated_groups[fi_val]
    if len(group) == 1:
        ax.plot([x_min, x_max], [group['y'].iloc[0]]*2, 'b-', linewidth=0.5, alpha=0.3)
    else:
        x_smooth = np.linspace(group['x'].min(), group['x'].max(), 100)
        ax.plot(x_smooth, fi_interpolators[fi_val](x_smooth), 'b-', linewidth=0.5, alpha=0.3)

for h_val in h_values_available:
    y_min_h, y_max_h = h_to_y_range[h_val]
    ax.plot([h_to_x[h_val]]*2, [y_min_h, y_max_h], 'r-', linewidth=0.5, alpha=0.3)

# Рисуване на най-близките изолинии (удебелени)
for fi_val in [closest_fi_lower, closest_fi_upper]:
    if fi_val is not None:
        group = fi_aggregated_groups[fi_val]
        if len(group) == 1:
            ax.plot([x_min, x_max], [group['y'].iloc[0]]*2, 'b-', linewidth=2, alpha=0.7,
                   label=f'Fi={fi_val}' if fi_val in [closest_fi_lower, closest_fi_upper] else "")
        else:
            x_smooth = np.linspace(group['x'].min(), group['x'].max(), 100)
            ax.plot(x_smooth, fi_interpolators[fi_val](x_smooth), 'b-', linewidth=2, alpha=0.7,
                   label=f'Fi={fi_val}' if fi_val in [closest_fi_lower, closest_fi_upper] else "")

for h_val in [closest_h_lower, closest_h_upper]:
    if h_val is not None:
        y_min_h, y_max_h = h_to_y_range[h_val]
        ax.plot([h_to_x[h_val]]*2, [y_min_h, y_max_h], 'r-', linewidth=2, alpha=0.7,
               label=f'H={h_val}' if h_val in [closest_h_lower, closest_h_upper] else "")

# Маркиране на пресечната точка
ax.plot([x_h], [y_tau], 'ro', markersize=8, label=f'τb = {y_tau:.4f}')

# Настройки на графиката
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.set_xlabel('x', fontsize=12)
ax.set_ylabel('y', fontsize=12)
ax.set_title(f'Номограма Fi-H (Fi={fi_value}, H={h_value})', fontsize=14)
ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper left', bbox_to_anchor=(1, 1))

# Показване на резултата
st.success(f"Изчислена стойност на τb: `{y_tau:.6f}`")
st.pyplot(fig)

# Информация за интерполацията
st.divider()
st.subheader('Информация за интерполацията')

if fi_value in fi_values_available:
    st.write(f"- Fi={fi_value}: Точна стойност (съвпада с изолиния)")
else:
    st.write(f"- Fi={fi_value}: Интерполирана между {closest_fi_lower} и {closest_fi_upper}")

if h_value in h_values_available:
    st.write(f"- H={h_value}: Точна стойност (съвпада с изолиния)")
else:
    st.write(f"- H={h_value}: Интерполирана между {closest_h_lower} и {closest_h_upper}")

# Допълнителна информация
st.divider()
st.subheader('Технически параметри')
st.write(f"Обхват на Fi: {min_fi} до {max_fi}")
st.write(f"Обхват на H: {min_h} до {max_h}")
st.write(f"Брой Fi изолинии: {len(fi_values_available)}")
st.write(f"Брой H изолинии: {len(h_values_available)}")
