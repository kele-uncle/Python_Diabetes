const trigger = document.getElementById("js_navigation-collapse-trigger");
const navigationContainer = document.getElementById("js_navigation-container");
const navigationModifierClass = "navigation-container--collapsed";

const animationDuration = 400;

const widthChange = anime.timeline({
  autoplay: false,
  easing: "easeInOutBack",
});

widthChange
  .add({
    targets: navigationContainer,
    width: 80,
    duration: animationDuration,
  })
  .add(
    {
      targets: ".js_navigation-item-name",
      opacity: [1, 0],
      duration: animationDuration / 2,
    },
    `-=${animationDuration}`
  );

let navigationIsCollapsed = false;

trigger.addEventListener("click", function () {
  navigationIsCollapsed = !navigationIsCollapsed;

  if (navigationIsCollapsed) {
    navigationContainer.classList.add(navigationModifierClass);
  } else {
    navigationContainer.classList.remove(navigationModifierClass);
  }

  if (widthChange.began) {
    widthChange.reverse();
    if (widthChange.progress === 100 && widthChange.direction === "reverse") {
      widthChange.completed = false;
    }
  }

  if (widthChange.paused) {
    widthChange.play();
  }
});