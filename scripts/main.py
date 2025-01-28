import logging
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

# Добавляем логирование для более удобного отслеживания
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Класс для сбора и работы с данными
class HardwareMonitoringData:
    def __init__(self, file_path):
        self.file_path = file_path
        self.session_info = None
        self.gpu_info = None
        self.headers = None
        self.meta_data = []
        self.metrics_df = pd.DataFrame()
        self.rows_to_skip = 0

    def read_file(self, start_row=0):
        """
        Чтение и обработка файла.
        Заголовки, информация о GPU, метаданные (единицы измерения по оси Y,
        мин. и макс. значения для графика каждой метрики).
        """
        self.rows_to_skip = start_row

        try:
            with open(self.file_path, "r", encoding="latin1") as file:
                # Пропускаем строки если требуется начать не с начала файла, например у нас несколько сессий
                for _ in range(start_row):
                    file.readline()
                # Собираем данные в разные переменные и инкапсулируем, выводим в консоль информацию о GPU
                self.session_info = file.readline().strip().split(",")
                self.gpu_info = file.readline().strip().split(",")

                if start_row == 0:
                    for i, gpu in enumerate(self.gpu_info[2:]):
                        logging.info(f"GPU{i+1}: {gpu}")
                
                self.headers = [
                    header.strip() for header in file.readline().strip().split(",")
                ][2:]
                # Т.к. в файле hml MSI Afterburner колличество заголовков == колличеству следующих строк с метаданными,
                # мы можем расчитать номер строки с которой у нас начинаются данные со значениями для построения графиков.
                # Получается прибавляем длину заголовков +3 за каждую прочитанную ранее строчку.
                self.rows_to_skip += len(self.headers) + 3

                # Собираем наши метаданные
                for line in file:
                    data_line = [item.strip() for item in line.strip().split(",")]
                    if data_line[0] == "03":
                        self.meta_data.append(data_line)
                    else:
                        break
        except Exception as e:
            logging.error(f"Не удалось прочитать файл: {e}")
            raise

    def read_metrics(self, chunk_size=1):
        """Чтение метрик из файла (значений)"""
        try:
            for chunk in pd.read_csv(
                self.file_path,
                chunksize=chunk_size,
                header=None,
                encoding="latin1",
                skiprows=self.rows_to_skip,
            ):
                first_column_value = str(chunk.iloc[0, 0]).strip()

                # Если первое значение в столбце равно "0", значит мы дошли до края нашей сессии.
                if first_column_value == "0":
                    logging.info("Собраны данные о сессии.")
                    # Возвращаем номер строки новой сессии
                    return self.rows_to_skip
                else:
                    # Если не "0", выводим строку как есть
                    self.metrics_df = pd.concat(
                        [self.metrics_df, chunk], ignore_index=True
                    )
                    self.rows_to_skip += 1
        except Exception as e:
            logging.error(f"Не удалось загрузить метрики: {e}")
            raise

    def get_metadata(self, offset=0):
        """Получение метаданных"""
        if not self.meta_data:
            return None
        metric_title = self.headers[offset].strip()
        metric_units = self.meta_data[offset][3].strip()
        min_val = float(self.meta_data[offset][4])
        max_val = float(self.meta_data[offset][5])
        return metric_title, metric_units, min_val, max_val


# Класс для отрисовки наших графиков
class MetricPlotter:
    def __init__(self, metrics_df, title, units, min_val, max_val, offset, fig, row):
        self.metrics_df = metrics_df
        self.title = title
        self.units = units
        self.min_val = min_val
        self.max_val = max_val
        self.offset = offset
        self.fig = fig
        self.row = row

    def plot(self):
        """Отрисовка графиков"""

        # Столбец с датой, используем для оси X
        time_data = pd.to_datetime(self.metrics_df.iloc[:, 1], dayfirst=True).dt.time
        # Столбец со значением, используем для оси Y
        metric_val = self.metrics_df.iloc[:, 2 + self.offset]
        metric_val = pd.to_numeric(metric_val, errors="coerce")

        # Добавляем график на фигуру
        self.fig.add_trace(
            go.Scatter(
                x=time_data,
                y=metric_val,
                mode="lines",
                name=f"{self.title} ({self.units})",
            ),
            row=self.row,
            col=1,
        )
        # Отключаем отображение времени по оси X, для более аккуратного вида.
        # Время будет отображаться при ховере на графиком
        self.fig.update_xaxes(showticklabels=False, row=self.row, col=1)
        # Добавляем диапазон значений для каждой метрики из файла
        self.fig.update_yaxes(range=[self.min_val, self.max_val], row=self.row, col=1)


# Функция для отрисовки графиков для каждой сессии в файле
def plot_session(file_path, session_start_row, session_count):
    # Создаем объект для чтения данных
    data_reader = HardwareMonitoringData(file_path)
    data_reader.read_file(session_start_row)  # Чтение и обработка файла
    session_start_row = data_reader.read_metrics()  # Чтение метрик, начало новой сессии

    # Создаем одну фигуру с подграфиками (вертикальная компоновка)
    fig = make_subplots(
        rows=len(
            data_reader.headers
        ),  # Определяем количество строк в зависимости от количества метрик
        cols=1,
        vertical_spacing=0.004,  # Настраиваем вертикальный отступ между графиками, чтобы заголовки помещались
        subplot_titles=[
            f"{data_reader.headers[i]}" for i in range(len(data_reader.headers))
        ],
    )

    # Общие настройки фигуры
    fig.update_layout(
        title={
            "text": f"Метрики производительности. Сессия №{session_count} ({data_reader.session_info[1]})",
            "font": {"size": 34},
        },
        hovermode="x unified",
        height=8000,
        width=1500,
    )

    offset = 0
    for row in range(1, len(data_reader.headers) + 1):  # Для всех метрик
        # Получаем метаданные
        metric_title, metric_units, min_val, max_val = data_reader.get_metadata(
            offset=offset
        )

        # Создаем объект для построения графика
        plotter = MetricPlotter(
            data_reader.metrics_df,
            metric_title,
            metric_units,
            min_val,
            max_val,
            offset,
            fig,
            row,
        )
        # Отрисовываем
        plotter.plot()
        offset += 1

    # Сохраняем график в файл HTML
    output_file = f"session{session_count}.html"
    fig.write_html(output_file)
    logging.info("Графики сохранены в файл: %s", output_file)
    return session_start_row


if __name__ == "__main__":
    # Основной скрипт
    file_path = "data/HardwareMonitoring.hml"
    session_start_row = 0
    session_counter = 1

    # Обрабатываем все сессии. Когда файл кончается, скрипт останавливается
    while True:
        if session_start_row is None:
            logging.info(
                "Графики были сохранены для каждой сессии. Всего %d сессии.",
                session_counter - 1,
            )
            break
        session_start_row = plot_session(file_path, session_start_row, session_counter)
        session_counter += 1
