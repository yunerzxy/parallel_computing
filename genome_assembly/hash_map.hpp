#pragma once

#include <upcxx/upcxx.hpp>
#include "kmer_t.hpp"

struct HashMap {
  //std::vector <kmer_pair> data;
  //std::vector <int> used;
  std::vector<upcxx::global_ptr<kmer_pair>> data;
  std::vector<upcxx::global_ptr<int>> used;

  size_t my_size;
  size_t global_size;
  size_t n_proc;

  size_t size() const noexcept;

  HashMap(size_t size);

  // Most important functions: insert and retrieve
  // k-mers from the hash table.
  bool insert(const kmer_pair &kmer, upcxx::atomic_domain<int>& ad);
  bool find(const pkmer_t &key_kmer, kmer_pair &val_kmer);

  // Helper functions

  // Write and read to a logical data slot in the table.
  void write_slot(uint64_t slot, const kmer_pair &kmer);
  kmer_pair read_slot(uint64_t slot);

  // Request a slot or check if it's already used.
  bool request_slot(uint64_t slot, upcxx::atomic_domain<int>& ad);
  bool slot_used(uint64_t slot);

  int index(uint64_t slot);
  int offset(uint64_t slot);
};

HashMap::HashMap(size_t size) {
  n_proc = upcxx::rank_n();
  global_size = size;
  //std::cout << "global size " << size << std::endl;
  my_size = ceil(1.0 * size / n_proc);
  //std::cout << "my_size " << my_size << std::endl;
  data.resize(n_proc, nullptr);
  used.resize(n_proc, 0);

  for (int i = 0; i < n_proc; ++i) {
    if (upcxx::rank_me() == i) {
      data[i] = upcxx::new_array<kmer_pair>(my_size);
      used[i] = upcxx::new_array<int>(my_size);
    }
    data[i] = upcxx::broadcast(data[i], i).wait();
    used[i] = upcxx::broadcast(used[i], i).wait();
  }
}

bool HashMap::insert(const kmer_pair &kmer, upcxx::atomic_domain<int>& ad) {
  uint64_t hash = kmer.hash();
  uint64_t probe = 0;
  //uint64_t p = 1;
  bool success = false;
  do {
    uint64_t slot = (hash + probe++) % global_size;
    //uint64_t slot = (hash + probe) % global_size;
    //probe = p * p++; // quadratic probing
    success = request_slot(slot, ad);
    if (success) {
      write_slot(slot, kmer);
    }
  } while (!success && probe < global_size);
  return success;
}

bool HashMap::find(const pkmer_t &key, kmer_pair &val) {
  uint64_t hash = key.hash();
  uint64_t probe = 0;
  //uint64_t p = 1;
  bool success = false;
  do {
    uint64_t slot = (hash + probe++) % global_size;
    //probe = p * p++;
    if (slot_used(slot)) {
      val = read_slot(slot);
      if (val.kmer == key) {
        success = true;
      }
    }
  } while (!success && probe < global_size);
  return success;
}

bool HashMap::slot_used(uint64_t slot) {
  upcxx::future<int> l_used = upcxx::rget(used[index(slot)] + offset(slot));
  int res = l_used.wait();
  return res != 0;
}

void HashMap::write_slot(uint64_t slot, const kmer_pair &kmer) {
  if (slot >= global_size || slot < 0) 
    throw std::runtime_error("out of scope");
  upcxx::rput(kmer, data[index(slot)] + offset(slot)).wait();
}

kmer_pair HashMap::read_slot(uint64_t slot) {
  if (slot >= global_size || slot < 0) 
    throw std::runtime_error("out of scope");
  return upcxx::rget(data[index(slot)] + offset(slot)).wait();
}

bool HashMap::request_slot(uint64_t slot, upcxx::atomic_domain<int>& ad) {
  upcxx::future<int> l_used = ad.fetch_add(used[index(slot)] + offset(slot), 1, std::memory_order_relaxed);
  int res = l_used.wait();
  return res == 0;
}

size_t HashMap::size() const noexcept {
  return my_size;
}

int HashMap::index(uint64_t slot) {
  return floor(slot / size());
}

int HashMap::offset(uint64_t slot) {
  return slot % size();
}
