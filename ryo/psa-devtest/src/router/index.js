import { createRouter, createWebHashHistory } from "vue-router";
import Home from "../views/Home.vue";

const routes = [
  {
    path: "/",
    name: "Home",
    component: Home,
  },
  {
    path: "/assessment",
    name: "Assessment",
    component: () =>
      import("../views/Assessment.vue"),
  },
  {
    path: "/record",
    name: "Record",
    component: () =>
      import("../views/Record.vue"),
  },
  {
    path: "/ranking",
    name: "Ranking",
    component: () =>
      import("../views/Ranking.vue"),
  }
];

const router = createRouter({
  history: createWebHashHistory(),
  routes,
});

export default router;
