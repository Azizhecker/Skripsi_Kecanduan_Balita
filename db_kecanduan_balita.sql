-- phpMyAdmin SQL Dump
-- version 5.2.1
-- https://www.phpmyadmin.net/
--
-- Host: 127.0.0.1
-- Waktu pembuatan: 28 Agu 2026 pada 20.30
-- Versi server: 10.4.32-MariaDB
-- Versi PHP: 8.2.12

SET SQL_MODE = "NO_AUTO_VALUE_ON_ZERO";
START TRANSACTION;
SET time_zone = "+00:00";


/*!40101 SET @OLD_CHARACTER_SET_CLIENT=@@CHARACTER_SET_CLIENT */;
/*!40101 SET @OLD_CHARACTER_SET_RESULTS=@@CHARACTER_SET_RESULTS */;
/*!40101 SET @OLD_COLLATION_CONNECTION=@@COLLATION_CONNECTION */;
/*!40101 SET NAMES utf8mb4 */;

--
-- Database: `db_kecanduan_balita`
--

-- --------------------------------------------------------

--
-- Struktur dari tabel `admin`
--

CREATE TABLE `admin` (
  `id` int(11) NOT NULL,
  `username` varchar(50) NOT NULL,
  `password` varchar(255) NOT NULL,
  `nama` varchar(100) NOT NULL
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `admin`
--

INSERT INTO `admin` (`id`, `username`, `password`, `nama`) VALUES
(1, 'admin', 'admin123', 'Administrator Skripsi');

-- --------------------------------------------------------

--
-- Struktur dari tabel `riwayat_prediksi`
--

CREATE TABLE `riwayat_prediksi` (
  `id` int(11) NOT NULL,
  `skor_x1` float NOT NULL,
  `skor_x2` float NOT NULL,
  `skor_x3` float NOT NULL,
  `skor_x4` float NOT NULL,
  `hasil_prediksi` varchar(20) NOT NULL,
  `created_at` timestamp NOT NULL DEFAULT current_timestamp()
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;

--
-- Dumping data untuk tabel `riwayat_prediksi`
--

INSERT INTO `riwayat_prediksi` (`id`, `skor_x1`, `skor_x2`, `skor_x3`, `skor_x4`, `hasil_prediksi`, `created_at`) VALUES
(1, 4, 2, 3, 6, 'RENDAH', '2026-08-26 15:14:15'),
(2, 4, 6, 4, 7, 'SEDANG', '2026-08-26 16:42:46'),
(3, 4, 6, 4, 8, 'SEDANG', '2026-08-28 17:12:56'),
(4, 4, 6, 4, 8, 'SEDANG', '2026-08-28 18:03:31'),
(5, 5, 8, 3, 6, 'TINGGI', '2026-08-28 18:18:53');

--
-- Indexes for dumped tables
--

--
-- Indeks untuk tabel `admin`
--
ALTER TABLE `admin`
  ADD PRIMARY KEY (`id`),
  ADD UNIQUE KEY `username` (`username`);

--
-- Indeks untuk tabel `riwayat_prediksi`
--
ALTER TABLE `riwayat_prediksi`
  ADD PRIMARY KEY (`id`);

--
-- AUTO_INCREMENT untuk tabel yang dibuang
--

--
-- AUTO_INCREMENT untuk tabel `admin`
--
ALTER TABLE `admin`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=2;

--
-- AUTO_INCREMENT untuk tabel `riwayat_prediksi`
--
ALTER TABLE `riwayat_prediksi`
  MODIFY `id` int(11) NOT NULL AUTO_INCREMENT, AUTO_INCREMENT=6;
COMMIT;

/*!40101 SET CHARACTER_SET_CLIENT=@OLD_CHARACTER_SET_CLIENT */;
/*!40101 SET CHARACTER_SET_RESULTS=@OLD_CHARACTER_SET_RESULTS */;
/*!40101 SET COLLATION_CONNECTION=@OLD_COLLATION_CONNECTION */;
